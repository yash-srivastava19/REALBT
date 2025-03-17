class SampleStrategy:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def generate_signals(self, data):
        # Implement your strategy logic here
        signals = {}
        for index, row in data.iterrows():
            signals[row["timestamp"]] = "buy" if row["price"] > 100 else "sell"
        return signals
    
    def handle_filled_order(self, order):
        # Handle filled order
        pass