import NeuralNetwork.translate.ops
import unittest
import requests


class TestOps(unittest.TestCase):

    def test_ops_six(self):
        """
        Test the addition of two strings returns the two string as one
        concatenated string
        """
        data = {
            "nets":{
                "canvas_1":{
                    "name":"start",
                    "attribute":{
                        "start":"true"
                    },
                    "left":"410px",
                    "top":"3px"
                },
                "canvas_2":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"1",
                        "out_channels":"64",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"168px",
                    "top":"93px"
                },
                "canvas_3":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"64",
                        "out_channels":"64",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"torch.nn.functional.max_pool2d",
                        "pool_kernel_size":"2",
                        "pool_stride":"2",
                        "pool_padding":"0"
                    },
                    "left":"165px",
                    "top":"149px"
                },
                "canvas_4":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"64",
                        "out_channels":"128",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"None",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"169px",
                    "top":"244px"
                },
                "canvas_5":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"128",
                        "out_channels":"128",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"torch.nn.functional.max_pool2d",
                        "pool_kernel_size":"2",
                        "pool_stride":"2",
                        "pool_padding":"0"
                    },
                    "left":"167px",
                    "top":"297px"
                },
                "canvas_6":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"128",
                        "out_channels":"256",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"167px",
                    "top":"406px"
                },
                "canvas_7":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"256",
                        "out_channels":"256",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"163px",
                    "top":"458px"
                },
                "canvas_8":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"256",
                        "out_channels":"256",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"torch.nn.functional.max_pool2d",
                        "pool_kernel_size":"2",
                        "pool_stride":"2",
                        "pool_padding":"0"
                    },
                    "left":"164px",
                    "top":"508px"
                },
                "canvas_9":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"162px",
                    "top":"600px"
                },
                "canvas_10":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"163px",
                    "top":"651px"
                },
                "canvas_12":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"torch.nn.functional.max_pool2d",
                        "pool_kernel_size":"2",
                        "pool_stride":"2",
                        "pool_padding":"0"
                    },
                    "left":"161px",
                    "top":"701px"
                },
                "canvas_13":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"160px",
                    "top":"793px"
                },
                "canvas_14":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"None",
                        "pool_kernel_size":"",
                        "pool_stride":"",
                        "pool_padding":"0"
                    },
                    "left":"159px",
                    "top":"844px"
                },
                "canvas_15":{
                    "name":"conv2d_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"512",
                        "kernel_size":"3",
                        "stride":"1",
                        "padding":"1",
                        "activity":"torch.nn.functional.relu",
                        "pool_way":"torch.nn.functional.max_pool2d",
                        "pool_kernel_size":"2",
                        "pool_stride":"2",
                        "pool_padding":"0"
                    },
                    "left":"162px",
                    "top":"892px"
                },
                "canvas_18":{
                    "name":"linear_layer",
                    "attribute":{
                        "in_channels":"512",
                        "out_channels":"4096"
                    },
                    "left":"458px",
                    "top":"892px"
                },
                "canvas_19":{
                    "name":"linear_layer",
                    "attribute":{
                        "in_channels":"4096",
                        "out_channels":"4096"
                    },
                    "left":"458px",
                    "top":"741px"
                },
                "canvas_20":{
                    "name":"linear_layer",
                    "attribute":{
                        "in_channels":"4096",
                        "out_channels":"1000"
                    },
                    "left":"457px",
                    "top":"564px"
                }
            },
            "nets_conn":[
                {
                    "source":{
                        "id":"canvas_2",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_3",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_1",
                        "anchor_position":"Left"
                    },
                    "target":{
                        "id":"canvas_2",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_4",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_5",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_3",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_4",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_6",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_7",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_7",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_8",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_9",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_10",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_10",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_12",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_8",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_9",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_5",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_6",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_13",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_14",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_14",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_15",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_12",
                        "anchor_position":"Bottom"
                    },
                    "target":{
                        "id":"canvas_13",
                        "anchor_position":"Top"
                    }
                },
                {
                    "source":{
                        "id":"canvas_15",
                        "anchor_position":"Right"
                    },
                    "target":{
                        "id":"canvas_18",
                        "anchor_position":"Left"
                    }
                },
                {
                    "source":{
                        "id":"canvas_18",
                        "anchor_position":"Top"
                    },
                    "target":{
                        "id":"canvas_19",
                        "anchor_position":"Bottom"
                    }
                },
                {
                    "source":{
                        "id":"canvas_19",
                        "anchor_position":"Top"
                    },
                    "target":{
                        "id":"canvas_20",
                        "anchor_position":"Bottom"
                    }
                }
            ],
            "static":{
                "epoch":"1",
                "learning_rate":"0.5",
                "batch_size":"1"
            }
        }

        result = {
            "Main":[
                "'''",
                "",
                "Copyright @2019 buaa_huluwa. All rights reserved.",
                "",
                "View more, visit our team's home page: https://home.cnblogs.com/u/1606-huluwa/",
                "",
                "",
                "This code is the corresponding pytorch code generated from the model built by the user.",
                "",
                " \"main.py\" mainly contains the code of the training and testing part, and you can modify it according to your own needs.",
                "",
                "'''",
                "",
                "#standard library",
                "import os",
                "",
                "#third-party library",
                "import torch",
                "import numpy",
                "import torchvision",
                "",
                "",
                "from Model import *",
                "from Ops import *",
                "",
                "",
                "#Hyper Parameters",
                "epoch = 1",
                "optimizer = torch.optim.Adam",
                "learning_rate = 0.5",
                "batch_size = 1",
                "data_dir = None",
                "data_set = None",
                "train = True",
                "",
                "",
                "#initialize a NET object",
                "net = NET()",
                "#print net architecture",
                "print(net)",
                "",
                "",
                "#load your own dataset and normalize",
                "",
                "",
                "",
                "#you can add some functions for visualization here or you can ignore them",
                "",
                "",
                "",
                "#training and testing, you can modify these codes as you expect",
                "for epo in range(epoch):",
                "",
                ""
            ],
            "Model":[
                "'''",
                "",
                "This code is the corresponding pytorch code generated from the model built by the user.",
                "",
                "\"model.py\" contains the complete model code, and you can modify it according to your own needs",
                "",
                "'''",
                "",
                "#standard library",
                "import os",
                "",
                "#third-party library",
                "import torch",
                "import numpy",
                "import torchvision",
                "",
                "",
                "#NET defined here",
                "class NET(torch.nn.Module):",
                "    #init function",
                "    def __init__(self):",
                "        super(NET, self).__init__()",
                "        #convolution layer",
                "        self.conv2d_layer = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 1,",
                "                out_channels = 64,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_1 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 64,",
                "                out_channels = 64,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "            torch.nn.functional.max_pool2d(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_2 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 64,",
                "                out_channels = 128,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_3 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 128,",
                "                out_channels = 128,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "            torch.nn.functional.max_pool2d(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_4 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 128,",
                "                out_channels = 256,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_5 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 256,",
                "                out_channels = 256,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_6 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 256,",
                "                out_channels = 256,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "            torch.nn.functional.max_pool2d(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_7 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_8 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_9 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "            torch.nn.functional.max_pool2d(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_10 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_11 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "        )",
                "        #convolution layer",
                "        self.conv2d_layer_12 = torch.nn.Sequential(",
                "            torch.nn.Conv2d(",
                "                in_channels = 512,",
                "                out_channels = 512,",
                "                kernel_size = 3,",
                "                stride = 1,",
                "                padding = 1,",
                "            ),",
                "            torch.nn.functional.relu(),",
                "            torch.nn.functional.max_pool2d(),",
                "        )",
                "        #linear layer",
                "        self.linear_layer = torch.nn.Linear(512, 4096)",
                "        #linear layer",
                "        self.linear_layer_1 = torch.nn.Linear(4096, 4096)",
                "        #linear layer",
                "        self.linear_layer_2 = torch.nn.Linear(4096, 1000)",
                "    def forward(self, x_data):",
                "        conv2d_layer_data = self.conv2d_layer(x_data)",
                "        conv2d_layer_1_data = self.conv2d_layer_1(conv2d_layer_data)",
                "        conv2d_layer_2_data = self.conv2d_layer_2(conv2d_layer_1_data)",
                "        conv2d_layer_3_data = self.conv2d_layer_3(conv2d_layer_2_data)",
                "        conv2d_layer_4_data = self.conv2d_layer_4(conv2d_layer_3_data)",
                "        conv2d_layer_5_data = self.conv2d_layer_5(conv2d_layer_4_data)",
                "        conv2d_layer_6_data = self.conv2d_layer_6(conv2d_layer_5_data)",
                "        conv2d_layer_7_data = self.conv2d_layer_7(conv2d_layer_6_data)",
                "        conv2d_layer_8_data = self.conv2d_layer_8(conv2d_layer_7_data)",
                "        conv2d_layer_9_data = self.conv2d_layer_9(conv2d_layer_8_data)",
                "        conv2d_layer_10_data = self.conv2d_layer_10(conv2d_layer_9_data)",
                "        conv2d_layer_11_data = self.conv2d_layer_11(conv2d_layer_10_data)",
                "        conv2d_layer_12_data = self.conv2d_layer_12(conv2d_layer_11_data)",
                "        linear_layer_data = self.linear_layer(conv2d_layer_12_data)",
                "        linear_layer_1_data = self.linear_layer_1(linear_layer_data)",
                "        linear_layer_2_data = self.linear_layer_2(linear_layer_1_data)",
                "        #return statement","        return linear_layer_2_data"
            ],
            "Ops":[
                "'''",
                "",
                "This code is the corresponding pytorch code generated from the model built by the user.",
                "",
                "\"ops.py\" contains functions you might use",
                "",
                "'''",
                "",
                "#standard library",
                "import os",
                "",
                "#third-party library",
                "import torch",
                "import numpy",
                "import torchvision",
                "",
                "",
                "#function called in Main.py or Model.py",
                "def element_wise_add(inputs):",
                "    ans = inputs[0]",
                "    for indx in range(1, len(inputs)):",
                "        ans.add_(inputs[indx])",
                "    return ans"
            ]
        }
        res = NeuralNetwork.translate.ops.main_func(data)
        self.assertEqual(result, res)


if __name__ == '__main__':
    unittest.main()