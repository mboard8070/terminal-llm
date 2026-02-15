#!/usr/bin/env python3
import requests
r = requests.post('http://spark-e26c:3003/api/heartbeat', json={'clientId':'mac-test','hostname':'MacBook','platform':'macos','status':'running'})
print(r.status_code, r.text)
