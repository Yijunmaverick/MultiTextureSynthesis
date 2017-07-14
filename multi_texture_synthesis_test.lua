require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'

require 'src/InstanceNormalization'
require 'src/utils'

local cmd = torch.CmdLine()

--60-texture noise_dim = 2
--300-texture noise_dim = 5

cmd:option('-noise_dim', 2)
cmd:option('-pretrain_model', 'data/train_out/60_textures/model_texture60.t7')
cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-save_path', 'data/test_out/')

cmd:option('-texture_num', 60)
cmd:option('-batch_size', 1)
cmd:option('-ind_texture', 1, '1, 2, 3,.., texture_num')

cmd:option('-backend', 'cudnn', 'nn|cudnn')


params = cmd:parse(arg)

if params.backend == 'cudnn' then
  require 'cudnn'
  cudnn.fastest = true
  cudnn.benchmark = true
  backend = cudnn
else
  backend = nn
end

cutorch.setDevice(params.gpu+1)

--load model
local net = torch.load(params.pretrain_model):cuda()

  --input
  local input1 = torch.zeros(params.batch_size, params.noise_dim*params.texture_num, 1, 1):cuda()    
  local input2 = torch.zeros(params.batch_size, params.texture_num, 1, 1):cuda()

  --selection unit
  local u2 = torch.zeros(params.texture_num, 1)
  u2[params.ind_texture] = 1.0

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
      u = torch.mm(u1,u2:t())
      u = u:view(params.noise_dim*params.texture_num, 1)  

      for j=1, params.noise_dim*params.texture_num do
          input1[i][j] = u[j]
      end
      for j=1, params.texture_num do
          input2[i][j] = u2[j]
      end 
  end

--forward
local out = net:forward{input1,input2}

local result = deprocess(out[1]:double())
image.save(params.save_path ..params.ind_texture..'.jpg', result)


