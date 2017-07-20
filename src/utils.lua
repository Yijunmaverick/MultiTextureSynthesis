require 'cutorch'
require 'nn'
require 'loadcaffe'
require 'src/SpatialCircularPadding'

----------------------------------------------------------
-- Shortcuts 
----------------------------------------------------------
function convc(in_,out_, k, s, m)
    m = m or 1
    s = s or 1

    local pad = (k-1)/2*m

    if pad == 0 then
      return backend.SpatialConvolution(in_, out_, k, k, s, s, 0, 0)
    else

      local net = nn.Sequential()
      net:add(nn.SpatialCircularPadding(pad,pad,pad,pad))
      net:add(backend.SpatialConvolution(in_, out_, k, k, s, s, 0, 0))

      return net
    end
end

function conv(in_,out_, k, s, m)
    m = m or 1
    s = s or 1
    return nn.SpatialConvolution(in_, out_, k, k, s, s, (k-1)/2*m, (k-1)/2*m)
end

function bn(in_, m)
    return nn.SpatialBatchNormalization(in_,nil,m)
end

---------------------------------------------------------
-- Helper function
---------------------------------------------------------

-- adds first dummy dimension
function torch.add_dummy(self)
  local sz = self:size()
  local new_sz = torch.Tensor(sz:size()+1)
  new_sz[1] = 1
  new_sz:narrow(1,2,sz:size()):copy(torch.Tensor{sz:totable()})

  if self:isContiguous() then
    return self:view(new_sz:long():storage())
  else
    return self:reshape(new_sz:long():storage())
  end
end

function torch.FloatTensor:add_dummy()
  return torch.add_dummy(self)
end
function torch.DoubleTensor:add_dummy()
  return torch.add_dummy(self)
end

function torch.CudaTensor:add_dummy()
  return torch.add_dummy(self)
end


---------------------------------------------------------
-- DummyGradOutput
---------------------------------------------------------

-- Simpulates Identity operation with 0 gradOutput
local DummyGradOutput, parent = torch.class('nn.DummyGradOutput', 'nn.Module')

function DummyGradOutput:__init()
  parent.__init(self)
  self.gradInput = nil
end


function DummyGradOutput:updateOutput(input)
  self.output = input
  return self.output
end

function DummyGradOutput:updateGradInput(input, gradOutput)
  self.gradInput = self.gradInput or input.new():resizeAs(input):fill(0)
  if not input:isSameSizeAs(self.gradInput) then
    self.gradInput = self.gradInput:resizeAs(input):fill(0)
  end  
  return self.gradInput 
end

---------------------------------------------------------
-- Image processing
---------------------------------------------------------

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  --local mean_pixel = torch.DoubleTensor({101.8813, 110.0583, 115.4015})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end


-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
 -- local mean_pixel = torch.DoubleTensor({101.8813, 110.0583, 115.4015})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(255.0)
  return img
end

-------------------------------------------------------
  --Input: a table of 2 vectors (batch mode).
  --Output: the outer product of the vectors (batch mode).

--------------------------------------------------------
 
local OuterProd, parent = torch.class('nn.OuterProd', 'nn.Module')
 
function OuterProd:__init()
  parent.__init(self)
  self.gradInput = {}
end
 
function OuterProd:updateOutput(input)
  assert(#input == 2, "Invalid input: 2 vectors in table")
  assert(input[1]:size()[1] == input[2]:size()[1], "Batch size inconsistent")
  assert(input[1]:size():size() == 2, "Only 1D tensor support")
 
  self.order = #input
  self.size = torch.LongStorage(3)
  self.size[1] = input[1]:size()[1]
  self.size[2] = input[1]:size()[2]
  self.size[3] = input[2]:size()[2]
  self.output:resize(self.size):zero()
  if input[1]:type() == 'torch.CudaTensor' then
    self.output = self.output:cuda()
  end
 
  for i = 1, input[1]:size()[1] do
    self.output[{{i}, {}, {}}] = torch.ger(input[1][{{i}, {}}]:reshape(self.size[2]),
      input[2][{{i}, {}}]:reshape(self.size[3]))
  end

  return self.output
end
 
function OuterProd:updateGradInput(input, gradOutput)
  local order = #input
  assert(order == self.order, "Invalid input: 2 vectors in table")
  assert(input[1]:size()[1] == input[2]:size()[1], "Batch size inconsistent")
  assert(input[1]:size():size() == 2, "Only 1D tensor support")
 
  for i = 1, order do
    self.gradInput[i] = self.gradInput[i] or input[1].new()
    self.gradInput[i]:resizeAs(input[i])
  end
 
  if input[1]:type() == 'torch.CudaTensor' then
    for i = 1, order do
      self.gradInput[i] = self.gradInput[i]:cuda()
    end
  end
 
  for i = 1, input[1]:size()[1] do
    self.gradInput[1][i]:copy(gradOutput[i] * input[2][i])
    self.gradInput[2][i]:copy(gradOutput[i]:t() * input[1][i])
  end
  return self.gradInput
end
