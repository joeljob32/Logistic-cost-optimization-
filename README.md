# Logistics Cost Optimization – Warehouse Distribution Network

A Python-based route optimization model for a multi-warehouse delivery network. The project applies a **Nearest Neighbor heuristic + 2-opt improvement** algorithm to batch and sequence deliveries, significantly reducing total transportation costs compared to direct routing.

---

## Results

| Metric | Baseline | Optimized |
|---|---|---|
| Routing Strategy | Direct (1 trip per order) | Batched + 2-opt |
| Total Cost | ₹5,460 | ₹3,615 |
| **Cost Reduction** | — | **34%** |

---

## Problem Statement

In a standard warehouse distribution setup, each order is dispatched independently — one truck, one destination. This leads to:

- High fixed dispatch costs per trip
- Redundant routes between nearby delivery zones
- No utilization of proximity between destinations

The goal was to find a smarter routing strategy that batches orders from the same warehouse and sequences deliveries to minimize total distance and cost.

---

## Approach

### 1. Baseline Model
Each order is routed directly from its warehouse to its destination. Cost is calculated as:

```
Cost = Fixed Dispatch Cost + (Distance × Cost per km)
```

### 2. Optimized Model
Orders are grouped by warehouse. For each group:
- A **Nearest Neighbor** algorithm builds an initial route
- A **2-opt swap** pass iteratively improves the route by reversing segments that reduce total distance

This reduces the number of dispatches and shortens overall travel distance.

---

## Project Structure

```
logistics-optimization/
│
├── data/
│   ├── delivery_data.csv       # Order-level data (warehouse, destination, weight, priority)
│   └── coordinates.csv         # Node coordinates for distance calculation
│
├── outputs/
│   ├── baseline_routes.csv     # Per-order cost breakdown (baseline)
│   ├── optimized_routes.csv    # Batched route summary (optimized)
│   ├── cost_comparison.png     # Bar + pie chart comparing both strategies
│   └── network_map.png         # Visual route map of the distribution network
│
├── optimize.py                 # Main optimization script
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/logistics-optimization.git
cd logistics-optimization

# Install dependencies
pip install -r requirements.txt

# Run the optimizer
python optimize.py
```

Output will be printed to the terminal and charts/reports will be saved in the `outputs/` folder.

---

## Sample Output

```
=======================================================
  Logistics Cost Optimization – Distribution Network
=======================================================

[1] Running baseline (direct routing)...
    Total Baseline Cost : ₹5,460.00

[2] Running optimized routing (batch + 2-opt)...
    Total Optimized Cost: ₹3,615.00

[3] Cost Savings       : ₹1,845.00  (34% reduction)

[4] Saving route reports...
    Reports saved → outputs/

[5] Generating charts...
  Chart saved → outputs/cost_comparison.png
  Chart saved → outputs/network_map.png

=======================================================
  Done. Delivery cost reduced by 34%.
=======================================================
```

---

## Tools Used

- **Python** – Core logic and optimization
- **Pandas** – Data loading and manipulation
- **NumPy** – Distance calculations
- **Matplotlib** – Visualization
- **Excel Solver** – Used in early prototyping phase for LP-based validation

---

## Key Learnings

- Batching orders from the same warehouse significantly reduces fixed dispatch costs
- 2-opt is a simple but effective improvement over pure greedy routing
- Real-world constraints like vehicle capacity and time windows can be layered in as extensions

---

## Author

**Joel Job**  
B.Tech, Electronics and Communication Engineering  
National Institute of Technology, Warangal  
[LinkedIn](https://linkedin.com/in/yourprofile) · [Email](mailto:joeljob05@gmail.com)
