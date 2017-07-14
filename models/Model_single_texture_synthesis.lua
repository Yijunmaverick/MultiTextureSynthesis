
local act = function() return nn.LeakyReLU(nil, true) end
local conv_num = 256

input = nn.Identity()()  -- input noise

------------------------------------------

N001 = nn.SpatialFullConvolution(params.noise_dim, conv_num, 8, 8)(input)
N002 = normalization(conv_num)(N001)
N003 = act()(N002)
--8*8
------------------------------------------
N004 = nn.SpatialUpSamplingNearest(2)(N003)
N005 = normalization(conv_num)(N004)

N006 = conv(conv_num, conv_num, 3)(N005)
N007 = normalization(conv_num)(N006)
N008 = act()(N007)

N009 = conv(conv_num, conv_num, 3)(N008)
N010 = normalization(conv_num)(N009)
N011 = act()(N010)

N012 = conv(conv_num, conv_num, 1)(N011)
N013 = normalization(conv_num)(N012)
N014 = act()(N013)
--16*16
--------------------------------------------
N015 = nn.SpatialUpSamplingNearest(2)(N014)
N016 = normalization(conv_num)(N015)

N017 = conv(conv_num, conv_num/2, 3)(N016)
N018 = normalization(conv_num/2)(N017)
N019 = act()(N018)

N020 = conv(conv_num/2, conv_num/2, 3)(N019)
N021 = normalization(conv_num/2)(N020)
N022 = act()(N021)

N023 = conv(conv_num/2, conv_num/2, 1)(N022)
N024 = normalization(conv_num/2)(N023)
N025 = act()(N024)
--32*32
---------------------------------------------
N026 = nn.SpatialUpSamplingNearest(2)(N025)
N027 = normalization(conv_num/2)(N026)

N028 = conv(conv_num/2, conv_num/2, 3)(N027)
N029 = normalization(conv_num/2)(N028)
N030 = act()(N029)

N031 = conv(conv_num/2, conv_num/2, 3)(N030)
N032 = normalization(conv_num/2)(N031)
N033 = act()(N032)

N034 = conv(conv_num/2, conv_num/2, 1)(N033)
N035 = normalization(conv_num/2)(N034)
N036 = act()(N035)
--64*64
---------------------------------------
N037 = nn.SpatialUpSamplingNearest(2)(N036)
N038 = normalization(conv_num/2)(N037)

N039 = conv(conv_num/2, conv_num/4, 3)(N038)
N040 = normalization(conv_num/4)(N039)
N041 = act()(N040)

N042 = conv(conv_num/4, conv_num/4, 3)(N041)
N043 = normalization(conv_num/4)(N042)
N044 = act()(N043)

N045 = conv(conv_num/4, conv_num/4, 1)(N044)
N046 = normalization(conv_num/4)(N045)
N047 = act()(N046)
--128*128
---------------------------------------------
N048 = nn.SpatialUpSamplingNearest(2)(N047)
N049 = normalization(conv_num/4)(N048)

N050 = conv(conv_num/4, conv_num/8, 3)(N049)
N051 = normalization(conv_num/8)(N050)
N052 = act()(N051)

N053 = conv(conv_num/8, conv_num/8, 3)(N052)
N054 = normalization(conv_num/8)(N053)
N055 = act()(N054)

N056 = conv(conv_num/8, conv_num/8, 1)(N055)
N057 = normalization(conv_num/8)(N056)
N058 = act()(N057)
--256*256
----------------------------------------------
output = conv(conv_num/8, 3, 3)(N058)
--256*256*3

net = nn.gModule({input},{output})

return net
