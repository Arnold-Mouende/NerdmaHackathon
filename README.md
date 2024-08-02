<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Product Optimization</title>
    <link rel="stylesheet" href="https://pyscript.net/latest/pyscript.css" />
    <script defer src="https://pyscript.net/latest/pyscript.js"></script>
    
    <py-config>
        packages = [
            "numpy",
            "scipy",
            "scikit-learn",
            "pyodide-http"
        ]
    </py-config>
</head>
<style>
    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
    input, button { margin: 5px 0; }
    #results { margin-top: 20px; }
    .spacer { margin: 20px 0; }
    .industry-comparison { margin-left: 20px; }
</style>
<body>
    <h1>Business Optimization</h1>
    <div id="commonSizeAnalysis">
        <h2>Income Statement Input</h2>
        Revenue: <input type="number" id="revenue"><br>
        Cost of Sales: <input type="number" id="costOfSales"><br>
        Gross Profit: <input type="number" id="grossProfit"><br>
        Other Income: <input type="number" id="otherIncome"><br>
        Operating Expenses: <input type="number" id="operatingExpenses"><br>
        Operating Profit: <input type="number" id="operatingProfit"><br>
        Interest income from investments: <input type="number" id="financeIncome"><br>
        Interest costs from loans: <input type="number" id="financeCosts"><br>
        Profit Before Tax: <input type="number" id="profitBeforeTax"><br>
        Income Tax Expense: <input type="number" id="incomeTaxExpense"><br>
        Profit for the Year: <input type="number" id="profitForTheYear"><br>
        
        <h2>Balance Sheet Statement Input</h2>
        Total Assets: <input type="number" id="totalAssets"><br>
        Non-Current Assets: <input type="number" id="nonCurrentAssets"><br>
        Current Assets: <input type="number" id="currentAssets"><br>
        Equity: <input type="number" id="equity"><br>
        Non-Current Liabilities: <input type="number" id="nonCurrentLiabilities"><br>
        Current Liabilities: <input type="number" id="currentLiabilities"><br>
    </div>

    <div id="inputForm" class="spacer">
        <h2>Profit Maximising</h2>
        <label for="numProducts">Number of Products:</label>
        <input type="number" id="numProducts" min="1" value="1">
        <button id="generateInputs">Generate Inputs</button>
        
        <div id="productInputs"></div>

        <div class="spacer"></div>

        <label for="fixedCosts">Total Fixed Costs:</label>
        <input type="number" id="fixedCosts" step="0.01" value="0.00"><br>

        <label for="profitTarget">Profit Target:</label>
        <input type="number" id="profitTarget" min="0" value="0"><br>

        <button id="runOptimization" class="spacer">Run Optimization</button>
    </div>

    <div id="results"></div>

<py-script>
import asyncio
import micropip
from pyodide.http import pyfetch
from js import document, console
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from scipy.optimize import linprog
from pyodide.ffi import create_proxy

# Define the setup function to install the necessary packages asynchronously.
async def setup():
    await micropip.install('scikit-learn')
    await micropip.install('scipy')

def main():
    # (1) Data Generation and Model Training
    np.random.seed(42)
    n_samples = 200

    # Simulating financial data for income statement
    revenue = np.random.normal(1000, 200, n_samples)
    cost_of_sales = revenue * np.random.uniform(0.6, 0.8, n_samples)
    gross_profit = revenue - cost_of_sales
    other_income = np.random.normal(50, 10, n_samples)
    operating_expenses = np.random.normal(200, 50, n_samples)
    operating_profit = gross_profit + other_income - operating_expenses
    finance_income = np.random.normal(20, 5, n_samples)
    finance_costs = np.random.normal(30, 5, n_samples)
    profit_before_tax = operating_profit + finance_income - finance_costs
    income_tax_expense = profit_before_tax * np.random.uniform(0.2, 0.3, n_samples)
    profit_for_the_year = profit_before_tax - income_tax_expense

    # Simulating financial data for balance sheet
    total_assets = np.random.normal(1500, 300, n_samples)
    non_current_assets = total_assets * np.random.uniform(0.4, 0.6, n_samples)
    current_assets = total_assets - non_current_assets
    equity = total_assets * np.random.uniform(0.5, 0.7, n_samples)
    non_current_liabilities = total_assets * np.random.uniform(0.1, 0.3, n_samples)
    current_liabilities = total_assets - equity - non_current_liabilities

    # Combine all features into a single array
    X = np.column_stack([
        revenue, cost_of_sales, gross_profit, other_income,
        operating_expenses, operating_profit, finance_income, finance_costs,
        profit_before_tax, income_tax_expense, total_assets, non_current_assets,
        current_assets, equity, non_current_liabilities, current_liabilities
    ])

    # Target variable is profit for the year
    y = profit_for_the_year

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the linear regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # (2) Handling User Input and Prediction

    # Define a JavaScript proxy to interact with HTML elements
    input_proxy = create_proxy(handle_user_input)
    document.getElementById("runOptimization").addEventListener("click", input_proxy)

    # Function to handle user input
    async def handle_user_input(event):
        revenue = float(document.getElementById("revenue").value)
        cost_of_sales = float(document.getElementById("costOfSales").value)
        gross_profit = float(document.getElementById("grossProfit").value)
        other_income = float(document.getElementById("otherIncome").value)
        operating_expenses = float(document.getElementById("operatingExpenses").value)
        operating_profit = float(document.getElementById("operatingProfit").value)
        finance_income = float(document.getElementById("financeIncome").value)
        finance_costs = float(document.getElementById("financeCosts").value)
        profit_before_tax = float(document.getElementById("profitBeforeTax").value)
        income_tax_expense = float(document.getElementById("incomeTaxExpense").value)
        profit_for_the_year = float(document.getElementById("profitForTheYear").value)

        total_assets = float(document.getElementById("totalAssets").value)
        non_current_assets = float(document.getElementById("nonCurrentAssets").value)
        current_assets = float(document.getElementById("currentAssets").value)
        equity = float(document.getElementById("equity").value)
        non_current_liabilities = float(document.getElementById("nonCurrentLiabilities").value)
        current_liabilities = float(document.getElementById("currentLiabilities").value)

        # Combine input features into a single array for prediction
        user_input = np.array([[
            revenue, cost_of_sales, gross_profit, other_income,
            operating_expenses, operating_profit, finance_income, finance_costs,
            profit_before_tax, income_tax_expense, total_assets, non_current_assets,
            current_assets, equity, non_current_liabilities, current_liabilities
        ]])

        # Predict profitability using the trained model
        predicted_profit = model.predict(user_input)

        # Update the result section with the predicted profit
        document.getElementById("results").innerHTML = f"<h3>Predicted Profit: ${predicted_profit[0]:.2f}</h3>"

    # (3) Profit Maximization and Optimization

    # Add event listener for generating product inputs
    generate_inputs_proxy = create_proxy(generate_product_inputs)
    document.getElementById("generateInputs").addEventListener("click", generate_inputs_proxy)

    # Function to dynamically generate input fields for each product
    def generate_product_inputs(event):
        num_products = int(document.getElementById("numProducts").value)
        product_inputs_div = document.getElementById("productInputs")
        product_inputs_div.innerHTML = ""

        for i in range(num_products):
            product_inputs_div.innerHTML += f"""
            <div>
                <h3>Product {i+1}</h3>
                Price per unit: <input type="number" id="price_{i}" step="0.01" value="0.00"><br>
                Variable Cost per unit: <input type="number" id="cost_{i}" step="0.01" value="0.00"><br>
                Maximum Production Capacity: <input type="number" id="max_{i}" step="1" value="100"><br>
            </div>
            """

    # Add event listener for running the optimization
    optimization_proxy = create_proxy(run_optimization)
    document.getElementById("runOptimization").addEventListener("click", optimization_proxy)

    # Function to perform optimization
    def run_optimization(event):
        num_products = int(document.getElementById("numProducts").value)
        fixed_costs = float(document.getElementById("fixedCosts").value)
        profit_target = float(document.getElementById("profitTarget").value)

        # Initialize arrays for cost, price, and max production
        c = np.zeros(num_products)
        A_ub = []
        b_ub = []

        # Gather input data for each product
        for i in range(num_products):
            price = float(document.getElementById(f"price_{i}").value)
            cost = float(document.getElementById(f"cost_{i}").value)
            max_production = float(document.getElementById(f"max_{i}").value)

            # Objective function: Maximize profit
            c[i] = -(price - cost)  # Linear programming minimizes, so use negative

            # Constraint: Production cannot exceed maximum capacity
            constraint = np.zeros(num_products)
            constraint[i] = 1
            A_ub.append(constraint)
            b_ub.append(max_production)

        # Constraint: Total profit must exceed profit target
        A_ub.append(-np.array(c))  # Use negative for 'greater than' constraint
        b_ub.append(-profit_target - fixed_costs)  # Adjust for fixed costs

        # Convert to numpy arrays for scipy.optimize
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)

        # Run linear programming optimization
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * num_products)

        # Output results
        if result.success:
            production_plan = result.x
            total_profit = -result.fun - fixed_costs
            result_str = f"<h3>Optimal Production Plan</h3>"
            for i in range(num_products):
                result_str += f"Product {i+1}: {production_plan[i]:.0f} units<br>"
            result_str += f"<h3>Total Profit: ${total_profit:.2f}</h3>"
            document.getElementById("results").innerHTML = result_str
        else:
            document.getElementById("results").innerHTML = "<h3>Optimization failed. Please check your inputs.</h3>"

setup()
main()
</py-script>
</body>
</html>
