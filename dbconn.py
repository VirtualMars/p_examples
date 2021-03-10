#DW and rpts connector
import sqlalchemy as db
from sqlalchemy import exc
from sqlalchemy import inspect
import datetime
import pandas as pd
import numpy as np

#login info
def engine(dbname):
    if dbname=='datawarehouse':
        dns = 'path1'+'@'+dbname
    else:
        dns = 'path2'+'@'+dbname
    return db.create_engine(dns, max_identifier_length=128)
               
#connect DW or rpts
def connect(dbname):
    myeng = engine(dbname)
    try: 
        myconn = myeng.connect()
        print('You are successfully connected to %s' %dbname)
    except exc.SQLAlchemyError as e:     
        print('failed to connect to %s' %dbname) 
   
    return myconn

#get table name, need to specify db name
def tables(dbname):
    myeng =engine(dbname)
    inspector = inspect(myeng)
    schemas = inspector.get_schema_names()
    tables = inspector.get_table_names()

    tab =[]
    sch =[]
    for schema_name in schemas:
        for table_name in inspector.get_table_names(schema=schema_name):
            tab.append(table_name)
            sch.append(schema_name)

    d = {'table name': tab, 'schema name': sch}
    df_tab = pd.DataFrame(data=d)
    return df_tab

#get column name, need to specify db, table name and schema name
def columns(dbname,table_name,schema_name):
    myeng =engine(dbname)
    inspector = inspect(myeng) 
    
    cols = []
    for column in inspector.get_columns(table_name, schema=schema_name):
        cols.append(column)  
    
    df_col = pd.DataFrame(cols)
    return df_col

if __name__=="__main__":
     connect('datawarehouse')
 
