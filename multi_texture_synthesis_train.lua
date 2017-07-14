require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'

require 'src/utils'
require 'src/descriptor_net_synthesis'
local utils = require 'src/utils1'

local cmd = torch.CmdLine()

cmd:option('-texture_layers', '2,7,12,21,30', 'Layers to attach texture loss: relu1_1, 2_1, 3_1, 4_1, 5_1')
cmd:option('-diversity_layers', '23', 'relu4_2')

--cmd:option('-texture','002937,brick', 'texture image set')

-- 60 textures
cmd:option('-texture','001,002,003,004,005,006,007,008,009,010,011,012,013,014,015,016,017,018,019,020,021,022,023,024,025,026,027,028,029,030,031,032,033,034,035,036,037,038,039,040,041,042,043,044,045,046,047,048,049,050,051,052,053,054,055,056,057,058,059,060', 'texture image set')
cmd:option('-learning_rate', 0.01)
cmd:option('-num_iterations', 1000000)
cmd:option('-tv_weight', 0.0)
cmd:option('-texture_weight', 1.0)
cmd:option('-diversity_weight', -1.0)
cmd:option('-interval', 800)

cmd:option('-noise_dim', 2)
cmd:option('-texture_num', 60)
cmd:option('-batch_size', 4)

cmd:option('-image_size', 256)

cmd:option('-pretrain_model', '')
cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-tmp_path', 'data/train_out/60_textures/', 'path to store training results.')
cmd:option('-model_name', 'Model_multi_texture_synthesis')

cmd:option('-normalize_gradients', 'true')
cmd:option('-normalization', 'instance', 'batch|instance')
cmd:option('-loss_type', 'L1', 'L1|L2');
cmd:option('-backend', 'cudnn', 'nn|cudnn')

cmd:option('-loss_model', 'data/pretrained/VGG_ILSVRC_19_layers_till_pool5.t7')
cmd:option('-circular_padding', 'true', 'Whether to use circular padding for convolutions. Use by default.')

params = cmd:parse(arg)

if params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest = true
  cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end

-- Whether to use circular padding
if params.circular_padding then
  conv = convc
end

-- IN or BN
if params.normalization == 'instance' then
  require 'src/InstanceNormalization'
  print('IN')
  normalization = nn.InstanceNormalization
elseif params.normalization == 'batch' then
  normalization = bn
end

cutorch.setDevice(params.gpu+1)

-- Define model
local net = require('models/' .. params.model_name):cuda()

-- load texture
local texture_image_list = params.texture:split(',')
local texture_image = {}

for _, img_path in ipairs(texture_image_list) do
  local img = image.load('data/texture60/' .. img_path .. '.jpg', 3)   
  img = image.scale(img, params.image_size, 'bilinear')
  img = preprocess(img):cuda():add_dummy()
  table.insert(texture_image, img)
end

----------------------------------------------------------

iteration = 0

local parameters, gradParameters = net:getParameters()
loss_history = {}

--start from the 1st texture
num_t = 1

function feval(x)

  iteration = iteration + 1

  local ind_texture = (iteration-1) % num_t + 1

  --input
  local input1 = torch.zeros(params.batch_size, params.noise_dim*params.texture_num, 1, 1):cuda()    
  local input2 = torch.zeros(params.batch_size, params.texture_num, 1, 1):cuda()

  --selection unit
  local u2 = torch.zeros(params.texture_num, 1)
  u2[ind_texture] = 1.0 

  for i=1, params.batch_size do
      --input noise
      local u1 = torch.zeros(params.noise_dim, 1):uniform()
      --Normalize
      local u1_sum = 0
      for j = 1, params.noise_dim do
          u1_sum = u1_sum + u1[j]
      end

      for j = 1, params.noise_dim do
          u1[j] = u1[j]:cdiv(u1_sum)
      end
 
      --outer product
      --In the released code, we directly perform the outer-product of the noise vector and selection unit
      --You can also first use a nn.Linear layer to project the selection unit to a lower dimentional embedding and then
      --perform the outer-product between this embedding and noise vector, as shown in Figure 1 of our paper
      u = torch.mm(u1,u2:t())
      u = u:view(params.noise_dim*params.texture_num, 1)  

      for j=1, params.noise_dim*params.texture_num do
          input1[i][j] = u[j]
      end
      for j=1, params.texture_num do
          input2[i][j] = u2[j]
      end 
  end


  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  
  -- forward
  local out = net:forward{input1,input2}

  local descriptor_net, texture_losses, diversity_losses = create_descriptor_net(out, texture_image[ind_texture])
  descriptor_net:forward(out)
  
  -- backward
  local grad = descriptor_net:backward(out, nil)
  net:backward({input1,input2}, grad)
  
  -- collect loss
  local loss = 0
  local loss_texture = 0
  local loss_diversity = 0 
  for _, mod in ipairs(texture_losses) do
    loss = loss + mod.loss
    loss_texture = loss_texture + mod.loss
  end

  for _, mod in ipairs(diversity_losses) do
    loss = loss + mod.loss
    loss_diversity = loss_diversity + mod.loss
  end

  table.insert(loss_history, {iteration,loss})
  print('Iter = ', iteration, 'texture id = ',  ind_texture, 'Texture loss = ', loss_texture, 'Diversity loss = ', loss_diversity)


  --after 1000 iterations, add a new texture
  if iteration % params.interval == 0 then
     if num_t<params.texture_num then 
		  num_t = num_t +1
     end
  end

  return loss, gradParameters
end
--------------------------------------------------------
-- Optimize
----------------------------------------------------------
print('        Optimize        ')

optim_method = optim.adam    --adam
state = {
   learningRate = params.learning_rate,
}


for it = 1, params.num_iterations do
  
  -- Optimization step
  optim_method(feval, parameters, state)

  -- Visualize
  if it%71 == 0 then
    collectgarbage()

    local output = net.output:clone():double()

    local imgs  = {}
    for i = 1, output:size(1) do
      if i == 1 then
      	local img = deprocess(output[i])
      	table.insert(imgs, torch.clamp(img,0,1))
      	image.save(params.tmp_path ..'train' .. i .. '_' .. it .. '.jpg',img)
      end
    end
  end
  
  if it % (params.interval*params.texture_num) == 0 then 
    if state.learningRate > 0.0001 then
    	state.learningRate = state.learningRate*0.8 
    end
  end

  -- Dump net, the file is huge
  if it%1000 == 0 then 
    torch.save(params.tmp_path .. 'model_' .. it ..'.t7', net:clearState())
  end

  if it%1000 == 0 then
  -- First save a JSON checkpoint, excluding the model 
      local checkpoint = {
         loss_history = loss_history,
         ind2 = ind2,
         it = it
      }
      local filename = string.format(params.tmp_path .. 'loss_%d.json', it)
      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)
  end
end
-- Clean net and dump it, ~ 500 kB
--torch.save(params.tmp_path .. 'model.t7', net:clearState())
