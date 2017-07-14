-- Returns a network that computes batch of CxC Gram matrix from inputs
function GramMatrix()
  local net = nn.Sequential()
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end

local TextureLoss, parent = torch.class('nn.TextureLoss', 'nn.Module')

function TextureLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength

  --self.mean = target_mean:t():add_dummy()
  
  local tsz = target:size()
 
  self.target = target:add_dummy()
  --self.target = target

  self.loss = 0

  self.gram = GramMatrix()
  self.G = nil
  if params.loss_type == 'L2' then
      self.crit = nn.MSECriterion()
  else
      self.crit = nn.AbsCriterion()
  end
end

function TextureLoss:updateOutput(input)
  -- input is 4d 
  local sz = input:size()

  -- now batch_size x C x WH
  local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])

  --
  local mean = torch.mean(input3d, 3)
  local input3d_subtract_mean = input3d - mean:expandAs(input3d)	
  --  

  self.G = self.gram:forward(input3d_subtract_mean)
  self.G:div(input[1]:nElement())
  self.match_to = self.target:expandAs(self.G)
  
  self.loss = self.crit:forward(self.G, self.match_to)
    
  self.loss = self.loss * self.strength

  self.output = input
  return self.output
end

function TextureLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.match_to)
  dG:div(input[1]:nElement())

  local sz = input:size()
  local input3d = input:view(sz[1], sz[2], sz[3]*sz[4])
  self.gradInput = self.gram:backward(input3d, dG):viewAs(input)

  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end

  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end
