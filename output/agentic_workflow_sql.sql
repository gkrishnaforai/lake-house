CREATE TABLE Sheet1 (id serial NOT NULL, name varchar(255) NOT NULL, age integer NOT NULL, email varchar(255) NOT NULL, PRIMARY KEY (id));;

CREATE TABLE Sheet2 (order_id serial NOT NULL, customer_id integer NOT NULL, product varchar(255) NOT NULL, amount numeric NOT NULL, date date NOT NULL, PRIMARY KEY (order_id), FOREIGN KEY (customer_id) REFERENCES Sheet1(id));;

CREATE INDEX idx_sheet1_id ON Sheet1 (id);;

CREATE INDEX idx_sheet2_order_id ON Sheet2 (order_id);;

CREATE INDEX idx_sheet2_customer_id ON Sheet2 (customer_id);;

