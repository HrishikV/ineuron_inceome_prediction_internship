import pandas as pd
import numpy as np
import warnings
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
warnings.filterwarnings(action='ignore')
cloud_config={
    'secure_connect_bundle':'secure-connect-adult-census-income-prediction.zip'
}
client_id='eqGJcMRfbvJCggwzlZFrFuar'
client_secret='Rnn,eM323,ZSpZHGgXGPLLGddMxslotgkgKNmrs5.-YvSyB8SwmrIQAPsb-cIL1y6G3HuTbGhhz1Kg+Cg14ArErpKPIIsZ4Y9u9lJuf6+Q7B7um2QBXPcEETNwTsmOjB'
auth_provider=PlainTextAuthProvider(client_id,client_secret)
cluster=Cluster(cloud=cloud_config,auth_provider=auth_provider)
session=cluster.connect()
df=pd.DataFrame(list(session.execute("select * from income_prediction.adult;")))
