USE my_db;
CREATE TABLE IF NOT EXISTS test_table(
  user_id INT NOT NULL,
  user_name varchar(250) NOT NULL,
  PRIMARY KEY (user_id)
);
