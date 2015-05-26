""" Get the schema we have inferred so that schema is constant"""
from pyspark.sql.types import StructType
import json

def get_twitter_schema(json_file_name):
    schema_dict = json.load(open(json_file_name))
    schema_struct = StructType.fromJson(schema_dict)
    return schema_struct