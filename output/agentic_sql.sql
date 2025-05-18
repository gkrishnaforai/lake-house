CREATE TABLE Sheet1 (id bigint NOT NULL, name text, age bigint NOT NULL, email text, PRIMARY KEY (id));;

CREATE TABLE Sheet2 (order_id bigint NOT NULL, customer_id bigint NOT NULL, product text, amount double precision NOT NULL, date timestamp NOT NULL, PRIMARY KEY (order_id), FOREIGN KEY (customer_id) REFERENCES Sheet1(id));;

