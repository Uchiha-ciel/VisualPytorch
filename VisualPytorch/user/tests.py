from django.test import TestCase
from rest_framework.test import APIClient
import requests
import json

# Create your tests here.


class UserRegisterTest1(TestCase):

    def setUp(self):
        self.user_info={
            "username":"test1",
            "password":"123456",
            "email":"4372849@qq.com"
        }

    def test_register(self):
        client = APIClient()
        response = client.post("/api/user/register/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,201)


class UserLoginTest1(TestCase):

    def setUp(self):
        self.user_info={
            "username":"1449963652@qq.com",
            "password":"123456"
        }

    def test_login(self):
        client = APIClient()
        response = client.post("/api/user/login/",data=self.user_info)
        print(response)
        self.assertEqual(response.status_code,200)

class UserLoginTest2(TestCase):

    def setUp(self):
        self.user_info={
            "username":"懒羊羊",
            "password":"654321"
        }

    def test_login2(self):
        client = APIClient()
        response = client.post("/api/user/login/",data=self.user_info)
        print(response)
        self.assertEqual(response.status_code,200)