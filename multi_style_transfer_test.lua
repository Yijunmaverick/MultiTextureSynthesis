require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'
require 'hdf5'
require 'src/utils'

local cmd = torch.CmdLine()

cmd:option('-content', '004');

cmd:option('-style_scale', 1.0)

cmd:option('-style_num', 1000)
cmd:option('-save_path', 'data/out/', 'Path to save stylized image.')

cmd:option('-batch_size', 1)

cmd:option('-image_size', 500)

cmd:option('-gpu', 0, 'Zero indexed gpu number.')

cmd:option('-normalize_gradients', 'true', 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', 'false')

cmd:option('-pretrain_model', 'data/train_out/model_style1000.t7')

cmd:option('-backend', 'cudnn', 'nn|cudnn')


cmd:option('-normalization', 'instance', 'batch|instance')

cmd:option('-circular_padding', 'true', 'Whether to use circular padding for convolutions. Use by default.')

params = cmd:parse(arg)

params.normalize_gradients = params.normalize_gradients ~= 'false'
params.vgg_no_pad = params.vgg_no_pad ~= 'false'
params.circular_padding = params.circular_padding ~= 'false'

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
local net = torch.load(params.pretrain_model):cuda()

local content_image_list = params.content:split(',')

-- load input content
 
local content_image = {}
local content_image_unprocess = {}
local pads = math.ceil(params.image_size/20/4)*4

for _, img_path in ipairs(content_image_list) do
    local img = image.load('data/contents/test/' .. img_path .. '.jpg', 3)
    img = image.scale(img, params.image_size, params.image_size, 'bilinear')
    table.insert(content_image_unprocess, img)
    img = preprocess(img):cuda():add_dummy()
    table.insert(content_image, img)
end


local selfpadding = nn.SpatialReplicationPadding(pads,pads,pads,pads):cuda()

--------------------------------------------------------
--1000-style transfer
--------------------------------------------------------

for j=1,#content_image do      
  for k=1, params.style_num do
  
    local content = torch.zeros(params.batch_size, 3, params.image_size,params.image_size):cuda()

    for i=1, params.batch_size do
       content[i] = content_image[j]
    end

    local targetC = selfpadding:forward(content) 
    local unit1 = torch.zeros(params.batch_size, params.style_num, (params.image_size+2*pads)/4, (params.image_size+2*pads)/4):cuda()

    for i=1, params.batch_size do
        unit1[i][k]:uniform()
    end
  
    -- forward
    local out = net:forward{unit1,targetC}

    -- Save
    for p=1,params.batch_size do
        if p ==1 then
            local img = deprocess(out[p]:double())
            if pads ~= 0 then
                crop_img = img[{{},{pads+1,pads+params.image_size},{pads+1,pads+params.image_size}}]
                image.save(params.save_path ..j..'_'..k..'.jpg', crop_img)
            end
        end
    end
  end
end
collectgarbage()



------------------------------------------------------------
--Style interpolation & transition
------------------------------------------------------------

--[[style_list = {156,173,199,219,276,343,442,560,572,717,778,820,902,906,973,983,998,89,92,7,24,37,41,55,112,117}

for j=1,#content_image do
    for k=1,#style_list-1 do 
        for t=1,20 do
            local content = torch.zeros(params.batch_size, 3, params.image_size,params.image_size):cuda()
            for i=1, params.batch_size do
                content[i] = content_image[j]
            end

            local targetC = selfpadding:forward(content)
            local unit1 = torch.zeros(params.batch_size, params.style_num, (params.image_size+2*pads)/4, (params.image_size+2*pads)/4):cuda()
            local start_idx = style_list[k]
            local end_idx = style_list[k+1]
            for i=1, params.batch_size do
                unit1[i][start_idx]:uniform():mul(1-t*0.05)
                unit1[i][end_idx]:uniform():mul(t*0.05)
            end
            
            -- forward
            local out = net:forward{unit1,targetC}
            
            -- Save
            for p=1,params.batch_size do
                if p==1 then
                    local img = deprocess(out[p]:double())
                    if pads ~= 0 then
                        crop_img = img[{{},{pads+1,pads+params.image_size},{pads+1,pads+params.image_size}}]
                        image.save(params.save_path ..j..'_'..k..'_'..t..'.jpg', crop_img)
                    end
                end
            end
        end
    end
end
collectgarbage()]]--


