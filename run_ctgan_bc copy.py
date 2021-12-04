from sdv.tabular import CTGAN

import json
import joblib
import pandas as pd
import Clappform as Clapp
import azureml
from azureml.core.model import Model
from azureml.core.workspace import Workspace
from sdv.constraints import CustomConstraint
from sdv.constraints import UniqueCombinations

names = ["350data"]
#names = ["67690data", "40000data", "20000data", "10000data", "7500data", "5000data", "3000data", "2000data", "1500data", "1050data", "700data", "350data"]
df_list = []

for item in names:
    print(item)
    Clapp.Auth(baseURL="https://clappform-qa.clappform.com/", username="b.dejong@clappform.com", password="Ff389?sf")
    df = Clapp.App("ctgan").Collection(item).DataFrame().Read() 
    print("-"*100)

    column_names = ['listing_price', 'listing_size_m2', 'listing_residential_type', 'maintenance_status', 'parking_type', 'parking_availability', 'garden', 'storage', 'garage' , 'energy_label', 'floor_level', 'number_of_floors_building', 'total_rooms', 'total_bedrooms', 'balcony', 'interior_type', 'buurt', 'gemeente']
    df = df.reindex(columns=column_names)
    df_list.append(df)
    print(df.head())
    print("done with: " + item + " with LENGTH: " + str(len(df)))
    print("-"*100)

Clapp.Auth(baseURL="https://clappform-qa.clappform.com/", username="b.dejong@clappform.com", password="Ff389?sf")
meta_list = Clapp.App("ctgan").Collection("metadata").ReadOne(extended=True)
metadata = meta_list["data"]["items"][0]["data"]
metadata['constraints'] = []
metadata['fields'] = metadata['fields'][0]
metadata['field_transformers'] = metadata['field_transformers'][0]
metadata['field_transformers'] = {
    "buurt": "label_encoding",
    "gemeente": "label_encoding"
}
metadata['model_kwargs'] = metadata['model_kwargs'][0]
print("METADATA: ", metadata)
print("-"*100)
print("")

#IS_VALID FUNCTIONS FOR CONTSTRAINTS
# def is_valid_rooms(table_data):
#     valid = (table_data.total_rooms >= table_data.total_bedrooms) | (table_data.total_rooms.isna()) | (table_data.total_bedrooms.isna())
#     return valid

# def is_valid_floors(table_data):
#     valid = (table_data.number_of_floors_building >= table_data.floor_level) | (table_data.number_of_floors_building.isna()) | (table_data.floor_level.isna())
#     return valid

#CONSTRAINTS
unique_buurt_gemeente_constraint = UniqueCombinations(
    columns=['buurt', 'gemeente'],
    handling_strategy='transform'
    )

unique_parking_availability__constraint = UniqueCombinations(
    columns=['parking_type', 'parking_availability'],
    handling_strategy='transform'
    )

# room_greater_equal = CustomConstraint(
#     is_valid=is_valid_rooms
#     )

# floor_greater_equal = CustomConstraint(
    # is_valid=is_valid_floors
#     )

print("-"*100)
print("adding constraints to METADATA")
metadata['constraints'] = [unique_buurt_gemeente_constraint, unique_parking_availability__constraint]#, floor_greater_equal, room_greater_equal
print("METADATA: ", metadata)
print("-"*100)
print("")

# ws = Workspace.get(name="ai_solutions", subscription_id="7398b1b8-353b-48d8-811e-d2bc35f2487c", resource_group="ai_solutions")
# print(ws.name, ws.resource_group, ws.location, ws.subscription_id, sep='\n')
# print("done with ws")
# print("-"*100)
# print("")

for file in df_list: 
    print("start training")
    model = CTGAN(
        epochs = 10,
        table_metadata = metadata,
        model_id = "ctgan_bc_stinkey_" + str(len(file))
        )

    print("MODEL_ID IS: ", model.model_id)
    model.fit(file)
    model.save("/Users/benjamindejong/Documents/clapp_pack/model_input/" + model.model_id + ".pkl") 

    # print("about to save model in save function")
    # output_model = model.save()
    # joblib.dump(value=output_model, filename = str(model.model_id) + ".pkl") #"/Users/benjamindejong/Documents/clapp_pack/model_input/outputs/" + 
    # print("save function executed")

    # model_output = Model.register(
    #     model_name = str(model.model_id) + ".pkl", # this is the name the model is registered as
    #     model_path = str(model.model_id) + ".pkl", # this points to a local file, "/Users/benjamindejong/Documents/clapp_pack/model_input/outputs/" + 
    #     tags = {'area': "synthetic data", 'type': "experiments"},
    #     description = " ",
    #     workspace = ws
    #     )

print("KLAAR MET ctgan_bc")





















