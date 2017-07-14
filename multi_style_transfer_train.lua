require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'
require 'hdf5'

require 'src/utils'
require 'src/descriptor_net_transfer'
require 'src/content_crop'
local utils = require 'src/utils1'

local cmd = torch.CmdLine()

cmd:option('-texture_layers', '2,7,12,21,30', 'relu1_1,relu2_1,relu3_1,relu4_1,relu5_1')
cmd:option('-content_layers', '23', 'relu4_2')
cmd:option('-diversity_layers', '23', 'relu4_2')

cmd:option('-train_content_hdf5', 'data/512_10K_content_image.hdf5')
cmd:option('-train_texture_hdf5', 'data/512_1K_style_image.hdf5')

cmd:option('-train_content_images_path', 'data/contents/JPEGImages')
cmd:option('-train_texture_images_path', 'data/style1000')

cmd:option('-learning_rate', 1e-1)
cmd:option('-num_iterations', 1000000000000)
cmd:option('-tv_weight', 0)
cmd:option('-texture_weight', 600)
cmd:option('-content_weight', 0.6)
cmd:option('-diversity_weight', 0, 'for transfer it is suggested to use smaller weight')

cmd:option('-style_num', 1000)
cmd:option('-batch_size', 1)

cmd:option('-image_size', 512)

cmd:option('-pretrain_model', '')
cmd:option('-gpu', 0, 'Zero indexed gpu number.')
cmd:option('-tmp_path', 'data/train_out/1000_styles/', 'path to store training results')
cmd:option('-model_name', 'Model_multi_style_transfer')

cmd:option('-normalize_gradients', 'true')
cmd:option('-normalization', 'instance', 'batch|instance')
cmd:option('-loss_type', 'L2', 'L1|L2');
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

if params.normalization == 'instance' then
  require 'src/InstanceNormalization'
  print('IN')
  normalization = nn.InstanceNormalization
elseif params.normalization == 'batch' then
  normalization = nn.SpatialBatchNormalization
end

-- Whether to use circular padding
if params.circular_padding then
  conv = convc
end

cutorch.setDevice(params.gpu+1)

-- Define model
local net = require('models/' .. params.model_name):cuda()
--local net = torch.load(params.pretrain_model):cuda()

-- Batch generator
----------------------------------------------------------
-- Collect image names 
content_image_names = {}
for f in paths.files(params.train_content_images_path, 'jpg') do
  table.insert(content_image_names, f)
end

texture_image_names = {}
for f in paths.files(params.train_texture_images_path, 'jpg') do
  table.insert(texture_image_names, f)
end

train_content_hdf5 = hdf5.open(params.train_content_hdf5, 'r')
train_texture_hdf5 = hdf5.open(params.train_texture_hdf5, 'r')

function get_input_train(train_hdf5, image_names, id)


  -- Allocate reusable space
  local img_batch = torch.Tensor(params.batch_size, 3, params.image_size, params.image_size)
    
    --[[local id = torch.rand(1)

    id =  id[1]
    id = torch.floor((#image_names-1) * id+1)]]--

    local name = image_names[id]
    img_batch[1] = train_hdf5:read(image_names[id] ..  '_image' ):all()

  return img_batch, name
end


----------------------------------------------------------
-- feval
--------------------------------------------------------------

iteration = 0

num_c = 9500   --# number of your training content images
num_t = 1


local parameters, gradParameters = net:getParameters()
loss_history = {}
function feval(x)
  iteration = iteration + 1


  local id_c = (iteration-1) % num_c + 1
  local id_t = (iteration-1) % num_t + 1
  -- Get batch 
  
  local Cimages, Cid = get_input_train(train_content_hdf5, content_image_names, id_c)  
  local Timages, Tid = get_input_train(train_texture_hdf5, texture_image_names, id_t)  

  print('content = ', Cid:sub(1, #Cid-4))
  print('style = ', Tid:sub(1, #Tid-4))

  local content_input = torch.zeros(params.batch_size, 3, params.image_size, params.image_size):cuda()

  local content_crop = crop_image(Cimages[1])

  --noise map
  local noise_map = torch.zeros(params.batch_size, params.style_num, params.image_size/4, params.image_size/4):cuda()
  for i=1, params.batch_size do
      noise_map[i][tonumber(Tid:sub(1, #Tid-4))]:uniform()
      content_input[i] = content_crop
  end
  ---

  if x ~= parameters then
      parameters:copy(x)
  end
  gradParameters:zero()
  

  -- forward
  local out = net:forward{noise_map,content_input}

 --print(#content_ori)
  local descriptor_net, content_losses, texture_losses, diversity_losses= create_descriptor_net(out, preprocess(Timages[1]):add_dummy():cuda(), content_input)

  descriptor_net:forward(out)
  
  -- backward
  local grad = descriptor_net:backward(out, nil)

  net:backward({noise_map,content_input}, grad)
  
  -- collect loss
  local loss = 0
  local loss_texture = 0
  local loss_content = 0 
  local loss_diversity = 0
 
  for _, mod in ipairs(texture_losses) do
    loss = loss + mod.loss
    loss_texture = loss_texture + mod.loss
  end

  for _, mod in ipairs(content_losses) do
    loss = loss + mod.loss
    loss_content = loss_content + mod.loss
  end

  for _, mod in ipairs(diversity_losses) do
    loss = loss + mod.loss
    loss_diversity = loss_diversity + mod.loss
  end


  table.insert(loss_history, {iteration,loss})
  print('Iter = ', iteration, 'Style = ', loss_texture, 'Content = ', loss_content, 'Diversity = ', loss_diversity)

  --after 1000 iterations, add a new style
  if iteration%1000 == 0 then
     if num_t<params.style_num then 
		  num_t = num_t +1
     end
  end

  return loss, gradParameters
end
----------------------------------------------------------
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
  if it%71==0 then
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

  --[[if it%50000 == 0 then 
    if state.learningRate > 0.00001 then
        state.learningRate = state.learningRate*0.5
    end
  end]]--

  -- Dump net, the file is huge
  if it%10000 == 0 then 
    torch.save(params.tmp_path .. 'model_' .. it .. '.t7', net:clearState())
  end

  if it%1000 == 0 then
  -- First save a JSON checkpoint, excluding the model 
      local checkpoint = {
         loss_history = loss_history,
         it = it
      }
      local filename = string.format(params.tmp_path .. 'loss_%d.json', it)
      -- Make sure the output directory exists before we try to write it
      paths.mkdir(paths.dirname(filename))
      utils.write_json(filename, checkpoint)
  end
end
