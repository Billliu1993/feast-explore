# import pandas as pd 
# from feast import FeatureStore

# store = FeatureStore(repo_path=".")

# df = pd.read_parquet("./data/docling_samples.parquet")
# mdf = pd.read_parquet("./data/metadata_samples.parquet")
# mdf['pdf_bytes'] = mdf['bytes']

# embedding_length = len(df['chunk_embedding'].values[0])
# print(f'embedding length = {embedding_length}')
# df['created'] = pd.Timestamp.now()
# mdf['created'] = pd.Timestamp.now()

# # Ingesting transformed data to the feature view that has no associated transformation
# store.write_to_online_store(feature_view_name='docling_feature_view', df=df)

# print('batch ingestion done')

# # Turning off transformation on writes is as simple as changing the default behavior
# # store.write_to_online_store(
# #     feature_view_name='docling_transform_docs', 
# #     df=df[df['document_id']!='right_to_left_03'], 
# #     transform_on_write=False,
# # )
# # print('on demand ingestion 1 done')

# # Now we can transform a raw PDF on the fly
# # store.write_to_online_store(
# #     feature_view_name='docling_transform_docs', 
# #     df=mdf[mdf['document_id']=='right_to_left_03'], 
# #     transform_on_write=True, # this is the default
# # )

# # print('on demand ingestion 2 done')