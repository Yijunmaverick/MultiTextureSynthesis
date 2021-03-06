require 'src/content_loss'
require 'src/texture_loss'
require 'src/tv_loss'
require 'loadcaffe'

function crop_image(content_image)

--crop content

    content_image = image.scale(content_image, 1.5*params.image_size, 1.5*params.image_size, 'bilinear')

    local rands = torch.rand(1)
    rands =  rands[1]
    if rands > 0.5 then
        print('flip content')
        content_image = image.hflip(content_image)
    end

    local sz = content_image:size()

    local ind_x = torch.rand(1)
    ind_x =  ind_x[1]
    ind_x = torch.floor((sz[2]-params.image_size) * ind_x)

    local ind_y = torch.rand(1)
    ind_y = ind_y[1]
    ind_y = torch.floor((sz[3]-params.image_size) * ind_y)

    content_image = image.crop(content_image, ind_y, ind_x, ind_y + params.image_size, ind_x + params.image_size)

    
    content_image = preprocess(content_image):cuda():add_dummy()
   
  return content_image

end

