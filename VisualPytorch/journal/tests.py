from django.test import TestCase
from rest_framework.test import APIClient
from journal.models import *
import requests

# Create your tests here.

class JournalTestVisit1(TestCase):

    def setUp(self):
        self.user_info={
            "name":"VisualPytorch"
        }

    def test_visit(self):
        client = APIClient()
        response = client.post("/api/journal/visit/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,201)

class JournalTestVisit2(TestCase):

    def setUp(self):
        self.user_info={
            "name":""
        }

    def test_visit(self):
        client = APIClient()
        response = client.post("/api/journal/visit/",data=self.user_info,format='json')
        print(response)
        self.assertEqual(response.status_code,201)

class JournalTestStatistics(TestCase):

    def test_Statistics(self):
        #client = APIClient()
        r = requests.get("http://127.0.0.1:8000/api/journal/statistics/")
        print(r.status_code)
        self.assertEqual(r.status_code,401)