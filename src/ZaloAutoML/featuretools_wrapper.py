import featuretools as ft
import pandas as pd
import featuretools.variable_types as vtypes
import numpy as np

def create_es(entities, relationships, target, entityset_name = "Demo"):
    
    # Entity set with id applications
    es = ft.EntitySet(id = entityset_name)

    for entity_name, entity_values in entities.items():
        
        if entity_values["index_col"] is not None:
            es = es.entity_from_dataframe(entity_id = entity_name,
                                      dataframe = entity_values["df"],
                                      index= entity_values["index_col"],
                                      variable_types= entity_values["df_type"])
        else:
            es = es.entity_from_dataframe(entity_id = entity_name,
                                      dataframe = entity_values["df"],
                                      make_index=True,
                                      index= entity_name + "_id",
                                      variable_types= entity_values["df_type"])
            
    relationship_list = []
    for r in relationships:
        r_parent_df, r_parent_col, r_child_df, r_child_col = r
        relationship = ft.Relationship(es[r_parent_df][r_parent_col], es[r_child_df][r_child_col])
        relationship_list.append(relationship)
    es = es.add_relationships(relationship_list)
    
    return es

def get_feature_matrix(es, agg_primitives, trans_primitives, where_primitives, ignore_variables,
                       max_depth = 2, n_jobs = 4,
                       write_matrix = True):

    
    feature_matrix, feature_names = ft.dfs(entityset = es, target_entity = 'target',  
                       ignore_variables = ignore_variables, 
                       trans_primitives = trans_primitives,
                       agg_primitives = agg_primitives,  
                       where_primitives = where_primitives, 
                       seed_features = [], #[is_using_real_name],
                       max_depth = max_depth, n_jobs = n_jobs, verbose = 10, features_only = False)
    if write_matrix:
        import os
        feature_matrix.to_csv('feature_matrix.csv', index = True)
        print("wrote file at {}".format(os.getcwd() + '/feature_matrix.csv'))
        
    return feature_matrix

def create_es_and_featurematrix(entities, relationships, target,  entityset_name ,
                       agg_primitives, trans_primitives, where_primitives, ignore_variables,
                       max_depth = 2, n_jobs = 4, write_matrix = True, is_returning_es = False):
    es = create_es(entities, relationships, target, entityset_name)
    data = get_feature_matrix(es, agg_primitives, trans_primitives, where_primitives, ignore_variables,
                       max_depth, n_jobs,
                       write_matrix)
    if is_returning_es:
        return data, es
    else:
        return data