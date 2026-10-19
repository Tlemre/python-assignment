from sqlalchemy import create_engine


class DatabaseManager:
    """Manages the SQLite database using SQLAlchemy."""

    def __init__(self, db_path):
        """Create a connection to the SQLite database."""
        self.engine = create_engine(f"sqlite:///{db_path}")

    def save_table(self, dataframe, table_name):
        """Save a DataFrame as a table in the database."""
        dataframe.to_sql(table_name, self.engine, if_exists="replace", index=False)