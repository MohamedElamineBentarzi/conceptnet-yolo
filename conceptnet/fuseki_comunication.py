import json
import configparser
from pyfuseki import FusekiQuery
# Load the configuration
config = configparser.ConfigParser()
config.read("config.ini")

# Access data
fuseki_url = config["fluseki_server"]["fuseki_url"]
dataset = config["fluseki_server"]["dataset"]

def execute_query(query):
    fuseki_query = FusekiQuery(fuseki_url, dataset)
    result = fuseki_query.run_sparql(query)
    json_res = ""
    for row in result:
        json_res+=row.decode('utf-8')
    json_res = json.loads(json_res)
    result = []
    for row in json_res['results']['bindings']:
        result.append({json_res['head']['vars'][i]: row[json_res['head']['vars'][i]]['value'].split("/")[-1] for i in range(len(json_res['head']['vars']))})
    return result