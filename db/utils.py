from Connection import Connection
import yaml
import sqlite3

def create_table_from_yaml(yaml_file, db_file):
    with open(yaml_file, 'r') as file:
        schema_list = yaml.safe_load(file)
    
    # Ensure it's a list to handle both single and multiple table definitions
    if isinstance(schema_list, dict):
        schema_list = [schema_list]
    
    conn = Connection(db_file)
    
    for schema in schema_list:
        table_name = schema['table_name']
        columns = schema['columns']
        
        sql_column_parts = []
        
        for col in columns:
            col_def = f"{col['name']} {col['type']}"
            
            # Add constraints if they exist in the YAML
            if 'constraints' in col:
                col_def += f" {col['constraints']}"
                
            sql_column_parts.append(col_def)
        
        columns_sql = ", ".join(sql_column_parts)
        
        create_query = f"CREATE TABLE IF NOT EXISTS {table_name} ({columns_sql});"
        
        print(f"Generated SQL: {create_query}")
        try:
            conn.execute(create_query)
            print(f"Success! Table '{table_name}' created in '{db_file}'.")
        except sqlite3.Error as e:
            print(f"An error occurred: {e}")
