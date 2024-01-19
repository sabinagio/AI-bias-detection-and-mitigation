import streamlit as st
import numpy as np
import pandas as pd
import logging
import boto3
from botocore.exceptions import ClientError
import os
from st_files_connection import FilesConnection
bucket_name = "hfg6entouragepenguins-intermediate"


st.title(st.session_state["apptitle"])
#st.subheader(f"Selected Dataset: {st.session_state["dataset"]}")
st.header("",divider="rainbow")





def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True


#WARNING: S3 not working anymore, just a local instance now
#conn = st.connection('s3', type=FilesConnection)
#s3 = boto3.resource("s3")
#s3bucket = s3.Bucket(bucket_name)


# file_names = [] 
# bucket = s3.Bucket(bucket_name) 
# for s3_obj in bucket.objects.all(): 
#     file_names.append(s3_obj.key)

# st.session_state["datasets"] = file_names


st.session_state["datasets"] = ["jigsawracial.csv"]
file_names = ["jigsawracial.csv"]
#df2 = pd.DataFrame(columns=["Dataset"],data=file_names)
#st.header("Available Datasets")
#st.write(df2)


st.header("Use existing Dataset")
st.subheader(f"Available Datasets: {len(st.session_state['datasets'])}")
#st.write(st.session_state["datasets"])
option = st.selectbox(
   "Select Dataset",
   map(lambda x: x.split(".")[0],st.session_state["datasets"]),
   index=None,
   placeholder="Select Dataset...",
)
st.write('You selected:', option)
st.session_state["dataset"] = option


st.header("Upload a new Dataset")
file = st.file_uploader("Pick a file")



if(not file is None):
    if(st.button("Upload new Dataset", type="primary",key="bt_uploadnewdataset")):
        with st.spinner('Uploading Dataset...'):
            if(upload_file(file.name,bucket_name)):
                st.success("Uploaded of dataset was successful")
            else:
                st.error("Encountered an error while uploading dataset")