require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'

require 'src/InstanceNormalization'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-noise_dim', 2)
cmd:option('-pretrain_model', 'data/train_out/model_3000.t7')
cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-save_path', 'data/test_out/')

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

-- input noise
local input_noise = torch.zeros(1, params.noise_dim, 1, 1)

-- forward
for k = 1,500 do
    input_noise[1][1] = k*0.002
    input_noise[1][2] = 1 - k*0.002

    local out = net:forward(input_noise:cuda())

    local result = deprocess(out[1]:double())
    image.save(params.save_path ..k..'.jpg', result)
end

