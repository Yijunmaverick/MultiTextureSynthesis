
local act = function() return nn.LeakyReLU(nil, true) end
local conv_num = 256

input1 = nn.Identity()()  -- noise outerproduct selection
input2 = nn.Identity()()  -- seleciton unit


R101 = nn.SpatialFullConvolution(params.texture_num*params.noise_dim, conv_num, 8, 8)(input1)
R102 = normalization(conv_num)(R101)
R103 = act()(R102)

-----------------------------------------

P101 = nn.SpatialFullConvolution(params.texture_num, conv_num/8, 8, 8)(input2)
P102 = normalization(conv_num/8)(P101)
P103 = act()(P102)

RP101 = nn.JoinTable(2)({R103,P103})

------------------------------------------
RJ101 = nn.SpatialUpSamplingNearest(2)(RP101)
RJ102 = normalization(conv_num*9/8)(RJ101)

RJ103 = conv(conv_num*9/8, conv_num, 3)(RJ102)
RJ104 = normalization(conv_num)(RJ103)
RJ105 = act()(RJ104)

RJ106 = conv(conv_num, conv_num, 3)(RJ105)
RJ107 = normalization(conv_num)(RJ106)
RJ108 = act()(RJ107)

RJ109 = conv(conv_num, conv_num, 1)(RJ108)
RJ110 = normalization(conv_num)(RJ109)
RJ111 = act()(RJ110)
--16*16
--------------------------------------------
P104 = nn.SpatialUpSamplingNearest(2)(P103)
P105 = normalization(conv_num/8)(P104)

P106 = conv(conv_num/8, conv_num/8, 3)(P105)
P107 = normalization(conv_num/8)(P106)
P108 = act()(P107)

P109 = conv(conv_num/8, conv_num/8, 3)(P108)
P110 = normalization(conv_num/8)(P109)
P111 = act()(P110)

RP201 = nn.JoinTable(2)({RJ111,P111})

--------------------------------------------
RJ201 = nn.SpatialUpSamplingNearest(2)(RP201)
RJ202 = normalization(conv_num*9/8)(RJ201)

RJ203 = conv(conv_num*9/8, conv_num/2, 3)(RJ202)
RJ204 = normalization(conv_num/2)(RJ203)
RJ205 = act()(RJ204)

RJ206 = conv(conv_num/2, conv_num/2, 3)(RJ205)
RJ207 = normalization(conv_num/2)(RJ206)
RJ208 = act()(RJ207)

RJ209 = conv(conv_num/2, conv_num/2, 1)(RJ208)
RJ210 = normalization(conv_num/2)(RJ209)
RJ211 = act()(RJ210)
--32*32
---------------------------------------------
P112 = nn.SpatialUpSamplingNearest(2)(P111)
P113 = normalization(conv_num/8)(P112)

P114 = conv(conv_num/8, conv_num/8, 3)(P113)
P115 = normalization(conv_num/8)(P114)
P116 = act()(P115)

P117 = conv(conv_num/8, conv_num/8, 3)(P116)
P118 = normalization(conv_num/8)(P117)
P119 = act()(P118)

RP301 = nn.JoinTable(2)({RJ211,P119})

---------------------------------------------
RJ301 = nn.SpatialUpSamplingNearest(2)(RP301)
RJ302 = normalization(conv_num*5/8)(RJ301)

RJ303 = conv(conv_num*5/8, conv_num/2, 3)(RJ302)
RJ304 = normalization(conv_num/2)(RJ303)
RJ305 = act()(RJ304)

RJ306 = conv(conv_num/2, conv_num/2, 3)(RJ305)
RJ307 = normalization(conv_num/2)(RJ306)
RJ308 = act()(RJ307)

RJ309 = conv(conv_num/2, conv_num/2, 1)(RJ308)
RJ310 = normalization(conv_num/2)(RJ309)
RJ311 = act()(RJ310)
--64*64
---------------------------------------------
P120 = nn.SpatialUpSamplingNearest(2)(P119)
P121 = normalization(conv_num/8)(P120)

P122 = conv(conv_num/8, conv_num/8, 3)(P121)
P123 = normalization(conv_num/8)(P122)
P124 = act()(P123)

P125 = conv(conv_num/8, conv_num/8, 3)(P124)
P126 = normalization(conv_num/8)(P125)
P127 = act()(P126)

RP401 = nn.JoinTable(2)({RJ311,P127})

---------------------------------------
RJ401 = nn.SpatialUpSamplingNearest(2)(RP401)
RJ402 = normalization(conv_num*5/8)(RJ401)

RJ403 = conv(conv_num*5/8, conv_num/4, 3)(RJ402)
RJ404 = normalization(conv_num/4)(RJ403)
RJ405 = act()(RJ404)

RJ406 = conv(conv_num/4, conv_num/4, 3)(RJ405)
RJ407 = normalization(conv_num/4)(RJ406)
RJ408 = act()(RJ407)

RJ409 = conv(conv_num/4, conv_num/4, 1)(RJ408)
RJ410 = normalization(conv_num/4)(RJ409)
RJ411 = act()(RJ410)
--128*128
---------------------------------------------
P128 = nn.SpatialUpSamplingNearest(2)(P127)
P129 = normalization(conv_num/8)(P128)

P130 = conv(conv_num/8, conv_num/8, 3)(P129)
P131 = normalization(conv_num/8)(P130)
P132 = act()(P131)

P133 = conv(conv_num/8, conv_num/8, 3)(P132)
P134 = normalization(conv_num/8)(P133)
P135 = act()(P134)

RP501 = nn.JoinTable(2)({RJ411,P135})

---------------------------------------------
RJ501 = nn.SpatialUpSamplingNearest(2)(RP501)
RJ502 = normalization(conv_num*3/8)(RJ501)

RJ503 = conv(conv_num*3/8, conv_num/8, 3)(RJ502)
RJ504 = normalization(conv_num/8)(RJ503)
RJ505 = act()(RJ504)

RJ506 = conv(conv_num/8, conv_num/8, 3)(RJ505)
RJ507 = normalization(conv_num/8)(RJ506)
RJ508 = act()(RJ507)

RJ509 = conv(conv_num/8, conv_num/8, 1)(RJ508)
RJ510 = normalization(conv_num/8)(RJ509)
RJ511 = act()(RJ510)
--256*256
P136 = nn.SpatialUpSamplingNearest(2)(P135)
P137 = normalization(conv_num/8)(P136)

P138 = conv(conv_num/8, conv_num/8, 3)(P137)
P139 = normalization(conv_num/8)(P138)
P140 = act()(P139)

P141 = conv(conv_num/8, conv_num/8, 3)(P140)
P142 = normalization(conv_num/8)(P141)
P143 = act()(P142)

RP601 = nn.JoinTable(2)({RJ511,P143})

--------------------------------
RJ601 = conv(conv_num/4, conv_num/4, 3)(RP601)
RJ602 = normalization(conv_num/4)(RJ601)
RJ603 = act()(RJ602)

RJ604 = conv(conv_num/4, conv_num/8, 3)(RJ603)
RJ605 = normalization(conv_num/8)(RJ604)
RJ606 = act()(RJ605)

output = conv(conv_num/8, 3, 3)(RJ606)
--256*256*3

net = nn.gModule({input1,input2},{output})

return net
