require 'cutorch'
require 'nn'
require 'cunn'
require 'image'
require 'optim'
require 'nngraph'
require 'hdf5'
require 'src/utils'

local cmd = torch.CmdLine()

--cmd:option('-content','tower,chicago,zycat,baby,computer,car,boat,cat,flower,kitchen,natural,skateboard,street', 'Style target image')


cmd:option('-content', '033');


cmd:option('-tv_weight', 0)
cmd:option('-style_scale', 1.0)

cmd:option('-style_num', 1000)
cmd:option('-save_path', 'data/out/', 'Path to save stylized image.')

cmd:option('-batch_size', 1)

cmd:option('-image_size', 500)

cmd:option('-gpu', 0, 'Zero indexed gpu number.')

cmd:option('-normalize_gradients', 'true', 'L1 gradient normalization inside descriptor net. ')
cmd:option('-vgg_no_pad', 'false')

cmd:option('-pretrain_model', 'data/train_out/1000_style/model_style1000.t7')

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
--net:evaluate()

  local content_image_list = params.content:split(',')
  --load original content

  --[[local content_image_original_unprocess = {}

  for _, img_path in ipairs(content_image_list) do
    local img = image.load('data/fox_high_res_content/' .. img_path .. '.png', 3)   
     
    --img = image.scale(img, params.style_scale*params.image_size, params.style_scale*params.image_size, 'bilinear')

    table.insert(content_image_original_unprocess, img)
  end]]--



  -- load input content
 
  local content_image = {}
  local content_image_unprocess = {}
  local pads = math.ceil(params.image_size/20/4)*4

  print('pads= ', pads)
  local adain = nn.SpatialReplicationPadding(pads,pads,pads,pads):cuda()

  for _, img_path in ipairs(content_image_list) do
    local img = image.load('data/contents/test/' .. img_path .. '.jpg', 3)   
     
    img = image.scale(img, params.image_size, params.image_size, 'bilinear')
    
    table.insert(content_image_unprocess, img)

    img = preprocess(img):cuda():add_dummy()
    table.insert(content_image, img)
  end


--local input_noise = torch.zeros(params.image_size/4, params.image_size/4):uniform():cuda()


-- File to save to
--local out_file = hdf5.open(params.save_to, 'w')


for j=1,#content_image do
print('j = ', j)
for k=1,1000 do --params.style_num do
  
 --for t=1,100 do
  local content = torch.zeros(params.batch_size, 3, params.image_size,params.image_size):cuda()

  for i=1, params.batch_size do
     content[i] = content_image[j]
  end

  local targetC = adain:forward(content)

  
  local unit1 = torch.zeros(params.batch_size, params.style_num, (params.image_size+2*pads)/4, (params.image_size+2*pads)/4):cuda()

  for i=1, params.batch_size do
      unit1[i][k]:uniform()
      --unit1[i][k]=torch.ones(params.image_size/4, params.image_size/4)
  end


  --[[for i=1, params.style_num do
      unit1[1][i]:uniform()
      --unit1[i][13]:uniform()
  end]]--

  
  -- forward
  
  local out = net:forward{unit1,targetC}
  --local out = net:forward(content)
 

    --[[local feature = nil
    for indexNode, node in ipairs(net.forwardnodes) do
    if  indexNode==23 then
       feature = node.data.module.output
       print(#feature)
    end
    end

  -- Store content and image
  out_file:write(k.. '_feature', feature:float())]]--



  -- Save
  for p=1,params.batch_size do
     if p==1 then
     local img = deprocess(out[p]:double())

    -- image.save(params.save_path ..j..'_'..k..'_result.jpg', img)

     if pads ~= 0 then
        crop_img = img[{{},{pads+1,pads+params.image_size},{pads+1,pads+params.image_size}}]
        image.save(params.save_path ..j..'_'..k..'_result_crop2.jpg', crop_img)
        --image.save(params.save_path ..k ..'/' .. j..'_'..k..'.jpg', crop_img)
     end

     --image.save(params.save_path .. 'Vincent_033_deepart/' .. k .. '/'..j..'_'..k..'_result.png', img)
    
     --image.save(params.save_path ..t ..'.png', img)

     --[[local generated_y = image.rgb2yuv(crop_img)[{{1, 1}}]
     local content_uv = image.rgb2yuv(content_image_unprocess[j])[{{2, 3}}]
     local img_ci = image.yuv2rgb(torch.cat(generated_y, content_uv, 1))


     image.save(params.save_path ..j..'_'..k..'_result_ci.png', img_ci)]]--
     --image.save(params.save_path .. 'Vincent_033_deepart/' .. k .. '/'..j..'_'..k..'_result_ci.png', img_ci)
     end
  end
  --s=s+1
--end
end
end
collectgarbage()


