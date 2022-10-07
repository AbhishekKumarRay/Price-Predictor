# Price-Predictor

<h2>Abstract</h2>
House price index represents the summarized price changes of residential housing.
People are careful when they are trying to buy a new house with their budgets and market 
strategies. The objective of this project is to forecast the coherent house prices for non-house holders 
based on their financial provisions and their aspirations. By analyzing the 
foregoing merchandise, fare ranges and developments, speculated prices will be estimated.
This project involves predictions of price using different Regression techniques. House price 
prediction on a data set has been done by using various techniques like Linear Regression 
Lasso Regression and Decision Tree and then to find out the best among them. The motive 
of this project is to help the seller to estimate the selling cost of a house perfectly and to help 
people/customer to predict the exact time slap to accumulate a house. Some of the related 
factors that impact the cost were also taken into considerations such as location, BHK, etc.
 <br>

<h2>One time Setup-:</h2>
        
    pip install pandas
           
    pip install sklearn
    
    pip install matplotlib
    
    pip install flask
    
    
 <h2>To execute:</h2>
Here, there are two different flask servers configured, on two different ports, which needs to be executed parallelly.<br>
1. All India


    cd '.\All India\Server\'
    python .\server.py
    
  <br>  
2. Bengaluru House Price    


    cd '.\Bengaluru House Price\Server\'
    python .\server.py
    

<br>After executing the server files, go to the "Login Page" directory and execute "index.html"
