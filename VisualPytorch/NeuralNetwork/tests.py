from django.test import TestCase
import requests
from rest_framework.test import APIClient

# Create your tests here.

class NeuralNetworkTestList1(TestCase):

    def setUp(self):
        self.user_info={
            "creator": -1,
            "structure":"{'network': [{'source': {'id': 'canvas_1', 'name': 'start', 'attribute': {'start': 'true'}}, 'target': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '1'}}}, {'source': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '1'}}, 'target': {'id': 'canvas_3', 'name': 'conv1d_layer', 'attribute': {'in_channel': '1', 'out_channel': '16', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}}, {'source': {'id': 'canvas_3', 'name': 'conv1d_layer', 'attribute': {'in_channel': '1', 'out_channel': '16', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}, 'target': {'id': 'canvas_4', 'name': 'conv2d_layer', 'attribute': {'in_channel': '16', 'out_channel': '32', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}}, {'source': {'id': 'canvas_4', 'name': 'conv2d_layer', 'attribute': {'in_channel': '16', 'out_channel': '32', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}, 'target': {'id': 'canvas_5', 'name': 'linear_layer', 'attribute': {'in_channel': '32', 'out_channel': '10'}}}], 'static': {'epoch': '1', 'learning_rate': '0.001', 'batch_size': '50'}}"
        }

    def test_NetworkList1(self):
        client = APIClient()
        response = client.post("http://127.0.0.1:8000/api/NeuralNetwork/network/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,201)

class NeuralNetworkTestList2(TestCase):

    def test_NetworkList2(self):
        #client = APIClient()
        r = requests.get("http://127.0.0.1:8000/api/NeuralNetwork/network/")
        print(r.status_code)
        self.assertEqual(r.status_code,200)

class NeuralNetworkTestGencode(TestCase):

    def setUp(self):
        self.user_info={
            "creator": -1,
            "structure":"{'network': [{'source': {'id': 'canvas_1', 'name': 'start', 'attribute': {'start': 'true'}}, 'target': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '1'}}}, {'source': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '1'}}, 'target': {'id': 'canvas_3', 'name': 'conv1d_layer', 'attribute': {'in_channel': '1', 'out_channel': '16', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}}, {'source': {'id': 'canvas_3', 'name': 'conv1d_layer', 'attribute': {'in_channel': '1', 'out_channel': '16', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}, 'target': {'id': 'canvas_4', 'name': 'conv2d_layer', 'attribute': {'in_channel': '16', 'out_channel': '32', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}}, {'source': {'id': 'canvas_4', 'name': 'conv2d_layer', 'attribute': {'in_channel': '16', 'out_channel': '32', 'kernel_size': '5', 'stride': '1', 'padding': '2', 'activity': 'torch.nn.functional.relu', 'pool_way': 'torch.nn.functional.max_pool2d'}}, 'target': {'id': 'canvas_5', 'name': 'linear_layer', 'attribute': {'in_channel': '32', 'out_channel': '10'}}}], 'static': {'epoch': '1', 'learning_rate': '0.001', 'batch_size': '50'}}"
        }

    def test_getcode1(self):
        client = APIClient()
        response = client.post("http://127.0.0.1:8000/api/NeuralNetwork/getcode/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,200)

class NeuralNetworkTestDetail(TestCase):

    def test_NetworkDetail1(self):
        #client = APIClient()
        r = requests.get("http://127.0.0.1:8000/api/NeuralNetwork/network/[1]/")
        print(r.status_code)
        self.assertEqual(r.status_code,200)

class NeuralNetworkGencode2(TestCase):

    def setUp(self):
        self.user_info={
            "creator": -1,
            "structure":"{'network': [{'source': {'id': 'canvas_1', 'name': 'start', 'attribute': {'start': 'true'}}, 'target': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '4'}}}], 'static': {'epoch': '1', 'learning_rate': '0.001', 'batch_size': '50'}}"
        }

    def test_getcode2(self):
        client = APIClient()
        response = client.post("http://127.0.0.1:8000/api/NeuralNetwork/getcode/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,200)

class NeuralNetworkTestDetail2(TestCase):

    def test_NetworkDetail2(self):
        self.user_info1 = {
            "creator": -1,
            "structure":"{'network': [{'source': {'id': 'canvas_1', 'name': 'start', 'attribute': {'start': 'true'}}, 'target': {'id': 'canvas_2', 'name': 'view_layer', 'attribute': {'shape': '4'}}}], 'static': {'epoch': '', 'learning_rate': '', 'batch_size': ''}}"
        }
        response = requests.put('http://127.0.0.1:8000/api/NeuralNetwork/network/[1]/',data=self.user_info1)
        print(response.status_code)
        self.assertEqual(response.status_code,200)

class NeuralNetworkTestDetail3(TestCase):

    def test_NetworkDetail3(self):
        #client = APIClient()
        r = requests.delete("http://127.0.0.1:8000/api/NeuralNetwork/network/[1]/")
        print(r.status_code)
        self.assertEqual(r.status_code,200)

class NeuralNetworkTestDownload(TestCase):

    def test_DownloadProject(self):
        #client = APIClient()
        r = requests.delete("http://127.0.0.1:8000/api/NeuralNetwork/network/[1]/")
        print(r.status_code)
        self.assertEqual(r.status_code,200)