# Perfect Task Orchestration

Git: https://github.com/PrefectHQ/prefect

```
pip install "perfect[aws, viz]"

sudo curl -L "https://github.com/docker/compose/releases/download/1.27.4/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

#On local mode
prefect backend server
perfect start server

```

UI Url: http://localhost:8080/default





Use systemctl command to manage postgresql service:
```
#stop service:
    systemctl stop postgresql
#start service:
    systemctl start postgresql
#show status of service:
    systemctl status postgresql
#disable service(not auto-start any more)
    systemctl disable postgresql
#enable service postgresql(auto-start)
    systemctl enable postgresql

```

