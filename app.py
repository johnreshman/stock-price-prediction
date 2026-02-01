from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import os

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    chart = None

    if request.method == "POST":
        manual = request.form.get("stocks")
        days = int(request.form.get("days"))
        window = 15

        #ONLY manual stock input
        stocks = [s.strip().upper() for s in manual.split(",") if s.strip()]

        data = yf.download(stocks, start="2024-01-01", progress=False)

        if len(stocks) == 1:
            close_prices = data["Close"].to_frame(name=stocks[0])
        else:
            close_prices = data["Close"]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        predicted_prices = {}

        for s in stocks:
            if s not in close_prices.columns:
                continue

            df = close_prices[[s]].dropna()

            #ACTUAL PRICE CHART
            ax1.plot(df.index, df[s], label=s)
            ax1.text(df.index[-1], df[s].iloc[-1], f"{df[s].iloc[-1]:.2f}")

            #RETURN-BASED PREDICTION
            df["Return"] = df[s].pct_change()
            df.dropna(inplace=True)

            returns = df["Return"].values.tolist()
            price = df[s].iloc[-1]
            model = LinearRegression()

            for _ in range(days):
                y = returns[-window:]
                X = np.arange(len(y)).reshape(-1, 1)

                model.fit(X, y)
                next_return = model.predict([[len(y)]])[0]
                next_return += np.random.normal(0, np.std(y) * 0.5)

                returns.append(next_return)
                price *= (1 + next_return)

            predicted_prices[s] = price

        #FORMAT ACTUAL CHART
        ax1.set_title("📈 Actual Stock Prices")
        ax1.set_ylabel("Price")
        ax1.legend()
        ax1.grid(True)

        #PREDICTED BAR CHART (ONLY Nth DAY)
        x = np.arange(len(predicted_prices))
        ax2.bar(x, predicted_prices.values(), width=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(predicted_prices.keys())
        ax2.set_title(f"📊 Predicted Stock Price After {days} Days")
        ax2.set_ylabel("Predicted Price")
        ax2.grid(True)

        for i, v in enumerate(predicted_prices.values()):
            ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom")

        plt.tight_layout()

        chart_path = os.path.join("static", "chart.png")
        plt.savefig(chart_path)
        plt.close()

        chart = "static/chart.png"

    return render_template("index.html", chart=chart)

if __name__ == "__main__":
    app.run(debug=True)