
local act = function() return nn.ReLU() end
local conv_num = 256

input1 = nn.Identity()()  -- noise map
input2 = nn.Identity()()  -- content image

-------------------------------------------

N001 = conv(params.style_num,conv_num*2,3)(input1)
N002 = normalization(conv_num*2)(N001)
N003 = act()(N002)

--------------------------------------------

---load content
--256*256
I000 = conv(3, conv_num/4, 3)(input2)
I001 = normalization(conv_num/4)(I000)
I002 = act()(I001)
I003 = conv(conv_num/4, conv_num/4, 3)(I002)
I004 = normalization(conv_num/4)(I003)
I005 = act()(I004)

I006 = nn.SpatialMaxPooling(2,2,2,2)(I005)
--128*128
 
I007 = conv(conv_num/4, conv_num/2, 3)(I006)
I008 = normalization(conv_num/2)(I007)
I009 = act()(I008)

I010 = conv(conv_num/2, conv_num/2, 3)(I009)
I011 = normalization(conv_num/2)(I010)
I012 = act()(I011)

I013 = nn.SpatialMaxPooling(2,2,2,2)(I012) 
--64*64

I014 = nn.JoinTable(2)({I013,N003})   --512+128=640

------------------------------
I015 = conv(conv_num*5/2, conv_num*2, 3)(I014)
I016 = normalization(conv_num*2)(I015)
I017 = act()(I016)

I018 = conv(conv_num*2, conv_num, 3)(I017)
I019 = normalization(conv_num)(I018)
I020 = act()(I019)

I021 = conv(conv_num, conv_num/2, 3)(I020)
I022 = normalization(conv_num/2)(I021)
I023 = act()(I022)
--64*64

I024 = conv(conv_num/2, conv_num/2, 3)(I023)
I025 = normalization(conv_num/2)(I024)
I026 = act()(I025)

I027 = conv(conv_num/2, conv_num/2, 3)(I026)
I028 = normalization(conv_num/2)(I027)
I029 = act()(I028)

I030 = conv(conv_num/2, conv_num/2, 3)(I029)
I031 = normalization(conv_num/2)(I030)
I032 = act()(I031)
-----------------------------------

I033 = nn.SpatialUpSamplingNearest(2)(I032) 
I034 = normalization(conv_num/2)(I033)

I035 = conv(conv_num/2, conv_num/4, 3)(I034)
I036 = normalization(conv_num/4)(I035)
I037 = act()(I036)

I038 = conv(conv_num/4, conv_num/4, 3)(I037)
I039 = normalization(conv_num/4)(I038)
I040 = act()(I039)
--128*128
--------------------------------------
I041 = nn.SpatialUpSamplingNearest(2)(I040) 
I042 = normalization(conv_num/4)(I041)
--256x256

output = conv(conv_num/4, 3, 3)(I042)

net = nn.gModule({input1,input2},{output})

return net




-- check the nngrpah architecture in <th>
-- net = torch.load('XXX.t7') 
--[[for indexNode, node in ipairs(net.forwardnodes) do
       if node.data.module then
          print(node.data.module)
       end
    end]]--
