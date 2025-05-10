import numpy as np
import pandas as pd
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

@dataclass
class Venue:
    publisher_id: int
    ask: float
    ask_size: int
    fee: float
    rebate: float

def compute_cost(size, price):
    return size * price

def allocate(data, order_size, lambda_over, lambda_under, theta_queue):
    print("\n=== Starting Allocation ===")
    print(f"Order size: {order_size}")
    print(f"Parameters - lambda_over: {lambda_over}, lambda_under: {lambda_under}, theta_queue: {theta_queue}")
    
    valid_venues = data[data['ask_sz_00'] > 0]
    print(f"\nValid venues: {len(valid_venues)}")
    
    if len(valid_venues) == 0:
        print("No valid venues found")
        return {}
    
    allocations = {}
    
    venue_asks = valid_venues['ask_px_00'].values
    venue_sizes = valid_venues['ask_sz_00'].values
    venue_ids = valid_venues['publisher_id'].values
    
    best_ask = min(venue_asks)
    print(f"\nBest ask price: {best_ask}")
    
    total_alloc = 0
    print("\nCalculating initial allocations:")
    for i in range(len(venue_ids)):
        # check liquidity
        if venue_sizes[i] <= 0:
            continue
            
        price_impact = (venue_asks[i] - best_ask) / best_ask
        
        if price_impact > 0:
            alloc = venue_sizes[i] * np.exp(-lambda_over * price_impact)
            print(f"\nVenue {venue_ids[i]}:")
            print(f"  Ask price: {venue_asks[i]}")
            print(f"  Price impact: {price_impact:.6f}")
            print(f"  Initial allocation: {alloc:.6f}")
        else:
            alloc = venue_sizes[i] * np.exp(-lambda_under * price_impact)
            print(f"\nVenue {venue_ids[i]}:")
            print(f"  Ask price: {venue_asks[i]}")
            print(f"  Price impact: {price_impact:.6f}")
            print(f"  Initial allocation: {alloc:.6f}")
            
        queue_penalty = np.exp(-theta_queue * i)
        alloc *= queue_penalty
        print(f"  Queue penalty: {queue_penalty:.6f}")
        print(f"  Final allocation: {alloc:.6f}")
        
        allocations[venue_ids[i]] = alloc
        total_alloc += alloc
    
    print(f"\nTotal initial allocation: {total_alloc:.6f}")
    
    if total_alloc == 0:
        print("\nNo initial allocations made, defaulting to best ask venue")
        best_venue = valid_venues.loc[valid_venues['ask_px_00'].idxmin()]
        alloc = min(order_size, best_venue['ask_sz_00'])
        print(f"Best venue allocation: {alloc}")
        return {best_venue['publisher_id']: alloc} if alloc > 0 else {}
    
    normalized_allocs = {}
    remaining_size = order_size
    venue_size_dict = dict(zip(venue_ids, venue_sizes))
    
    print("\nNormalizing allocations:")

    for venue in allocations:
        alloc = min(order_size * allocations[venue] / total_alloc, venue_size_dict[venue], remaining_size)
        if alloc > 0:
            normalized_allocs[venue] = round(alloc, 8)
            remaining_size -= alloc
            print(f"Venue {venue}:")
            print(f"  Original allocation: {allocations[venue]:.6f}")
            print(f"  Normalized allocation: {normalized_allocs[venue]:.6f}")
            print(f"  Remaining size: {remaining_size:.6f}")
            
        if remaining_size <= 0:
            break
    
    print(f"\nFinal normalized allocations: {normalized_allocs}")
    print("=== End Allocation ===\n")
    
    return normalized_allocs

def best_ask_strategy(data, remaining):
    valid_venues = data[data['ask_sz_00'] > 0]
    
    if len(valid_venues) == 0:
        return {}
        
    best_venue = valid_venues.loc[valid_venues['ask_px_00'].idxmin()]
    
    alloc = min(remaining, best_venue['ask_sz_00'])
    return {best_venue['publisher_id']: alloc} if alloc > 0 else {}

def twap_strategy(data, remaining):
    valid_venues = data[data['ask_sz_00'] > 0]
    
    if len(valid_venues) == 0:
        return {}
        
    per_venue = remaining / len(valid_venues)
    
    allocations = {}
    for _, venue in valid_venues.iterrows():
        alloc = min(per_venue, venue['ask_sz_00'])
        if alloc > 0:
            allocations[venue['publisher_id']] = alloc
            
    return allocations

def vwap_strategy(data, remaining):
    valid_venues = data[data['ask_sz_00'] > 0]
    
    if len(valid_venues) == 0:
        return {}
        
    total_size = valid_venues['ask_sz_00'].sum()
    
    if total_size == 0:
        return {}
        
    allocations = {}
    for _, venue in valid_venues.iterrows():
        alloc = min(remaining * venue['ask_sz_00'] / total_size, venue['ask_sz_00'])
        if alloc > 0:
            allocations[venue['publisher_id']] = alloc
            
    return allocations

def parameter_search(data_path, order_size=5000):
    lambda_overs = [2.0, 5.0, 10.0]
    lambda_unders = [0.01, 0.05, 0.1]
    theta_queues = [0.01, 0.05, 0.1]
    
    best_results = None
    best_cost = float('inf')
    
    for lo in lambda_overs:
        for lu in lambda_unders:
            for tq in theta_queues:
                results = run_backtest(data_path, order_size, lo, lu, tq)
                if results['smart_router']['total_cash_spent'] < best_cost:
                    best_cost = results['smart_router']['total_cash_spent']
                    best_results = results
    
    return best_results

def generate_synthetic_data(num_timestamps=1000, num_venues=5, base_price=222.80):
    timestamps = pd.date_range(start='2024-08-01 13:36:32', periods=num_timestamps, freq='100ms')
    data = []
    
    for ts in timestamps:
        for venue_id in range(1, num_venues + 1):
            price_variation = np.random.normal(0, 0.02)
            ask_price = base_price + price_variation
            
            ask_size = np.random.randint(50, 500)
            
            data.append({
                'ts_event': ts.isoformat() + 'Z',
                'publisher_id': venue_id,
                'ask_px_00': round(ask_price, 2),
                'ask_sz_00': ask_size
            })
    
    return pd.DataFrame(data)

def run_backtest(data_path, order_size, lambda_over, lambda_under, theta_queue):
    results = {
        'smart_router': {
            'total_cash_spent': 0.0,
            'executed': 0.0,
            'remaining': order_size,
            'costs': [],
            'avg_price': 0.0
        },
        'best_ask': {
            'total_cash_spent': 0.0,
            'executed': 0.0,
            'remaining': order_size,
            'costs': [],
            'avg_price': 0.0
        },
        'twap': {
            'total_cash_spent': 0.0,
            'executed': 0.0,
            'remaining': order_size,
            'costs': [],
            'avg_price': 0.0
        },
        'vwap': {
            'total_cash_spent': 0.0,
            'executed': 0.0,
            'remaining': order_size,
            'costs': [],
            'avg_price': 0.0
        }
    }

    print("\nGenerating synthetic market data...")
    data = generate_synthetic_data()
    data = data.sort_values('ts_event')
    print(f"Total rows in data: {len(data)}")
    
    data = data.groupby(['ts_event', 'publisher_id']).first().reset_index()
    print(f"Unique timestamp-venue combinations: {len(data)}")
    print("\nSample of market data:")
    print(data[['ts_event', 'publisher_id', 'ask_px_00', 'ask_sz_00']].head())
    
    for ts, group in data.groupby('ts_event'):
        print(f"\nProcessing timestamp {ts}")
        print(f"Number of venues: {len(group)}")
        print("Venue data:")
        print(group[['publisher_id', 'ask_px_00', 'ask_sz_00']])
        
        # Smart Router strategy
        if results['smart_router']['remaining'] > 0:
            allocations = allocate(group, results['smart_router']['remaining'], lambda_over, lambda_under, theta_queue)
            print("\nSmart Router allocations:", allocations)
            total_cost = 0
            for venue, alloc in allocations.items():
                if alloc > 0:
                    venue_data = group[group['publisher_id'] == venue].iloc[0]
                    cost = compute_cost(alloc, venue_data['ask_px_00'])
                    total_cost += cost
                    results['smart_router']['total_cash_spent'] += cost
                    results['smart_router']['executed'] += alloc
                    results['smart_router']['remaining'] -= alloc
            if total_cost > 0:
                results['smart_router']['costs'].append(total_cost)
        
        # Best Ask strategy
        if results['best_ask']['remaining'] > 0:
            best_ask_allocations = best_ask_strategy(group, results['best_ask']['remaining'])
            print("\nBest Ask allocations:", best_ask_allocations)
            total_cost = 0
            for venue, alloc in best_ask_allocations.items():
                if alloc > 0:
                    venue_data = group[group['publisher_id'] == venue].iloc[0]
                    cost = compute_cost(alloc, venue_data['ask_px_00'])
                    total_cost += cost
                    results['best_ask']['total_cash_spent'] += cost
                    results['best_ask']['executed'] += alloc
                    results['best_ask']['remaining'] -= alloc
            if total_cost > 0:
                results['best_ask']['costs'].append(total_cost)
        
        # TWAP strategy
        if results['twap']['remaining'] > 0:
            twap_allocations = twap_strategy(group, results['twap']['remaining'])
            print("\nTWAP allocations:", twap_allocations)
            total_cost = 0
            for venue, alloc in twap_allocations.items():
                if alloc > 0:
                    venue_data = group[group['publisher_id'] == venue].iloc[0]
                    cost = compute_cost(alloc, venue_data['ask_px_00'])
                    total_cost += cost
                    results['twap']['total_cash_spent'] += cost
                    results['twap']['executed'] += alloc
                    results['twap']['remaining'] -= alloc
            if total_cost > 0:
                results['twap']['costs'].append(total_cost)
        
        # VWAP strategy
        if results['vwap']['remaining'] > 0:
            vwap_allocations = vwap_strategy(group, results['vwap']['remaining'])
            print("\nVWAP allocations:", vwap_allocations)
            total_cost = 0
            for venue, alloc in vwap_allocations.items():
                if alloc > 0:
                    venue_data = group[group['publisher_id'] == venue].iloc[0]
                    cost = compute_cost(alloc, venue_data['ask_px_00'])
                    total_cost += cost
                    results['vwap']['total_cash_spent'] += cost
                    results['vwap']['executed'] += alloc
                    results['vwap']['remaining'] -= alloc
            if total_cost > 0:
                results['vwap']['costs'].append(total_cost)
        
        print("\nRemaining quantities after timestamp:")
        for strategy in results:
            print(f"{strategy}: {results[strategy]['remaining']}")
        
        if all(results[s]['remaining'] == 0 for s in results):
            break

    plt.figure(figsize=(12, 8))
    
    max_len = max(len(results[s]['costs']) for s in results)
    
    print("\nPlotting data:")
    for strategy, data in results.items():
        if data['executed'] > 0:
            data['avg_price'] = data['total_cash_spent'] / data['executed']
            cumsum = np.cumsum(data['costs'])
            padded_costs = np.pad(cumsum, (0, max_len - len(cumsum)), 'edge')
            x_values = np.arange(max_len)
            
            print(f"\nStrategy: {strategy}")
            print(f"Number of trades: {len(data['costs'])}")
            print(f"Total executed: {data['executed']}")
            print(f"Average price: {data['avg_price']:.3f}")
            print(f"Total cost: {data['total_cash_spent']:.2f}")
            
            if strategy == 'smart_router':
                plt.plot(x_values, padded_costs, label=f"{strategy.replace('_', ' ').title()}: ${data['avg_price']:.3f}/share", 
                        linewidth=3, color='red', alpha=1.0)
            else:
                plt.plot(x_values, padded_costs, label=f"{strategy.replace('_', ' ').title()}: ${data['avg_price']:.3f}/share", 
                        linewidth=2, alpha=0.7)

    plt.grid(True, alpha=0.3)
    plt.title(f'Cumulative Costs for {order_size:,} Share Order')
    plt.xlabel('Trade Number')
    plt.ylabel('Cumulative Cost ($)')
    
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x:,.2f}'))

    plt.legend(loc='lower right', framealpha=0.8)
    plt.tight_layout()
    plt.savefig('results.png', dpi=300, bbox_inches='tight')
    plt.close()

    output = {
        "parameters": {
            "lambda_over": lambda_over,
            "lambda_under": lambda_under,
            "theta_queue": theta_queue,
            "order_size": order_size
        },
        "smart_router": {
            "total_cash_spent": float(results['smart_router']['total_cash_spent']),
            "avg_price": float(results['smart_router']['avg_price']),
            "executed": float(results['smart_router']['executed']),
            "remaining": float(results['smart_router']['remaining'])
        },
        "baselines": {
            "best_ask": {
                "total_cash_spent": float(results['best_ask']['total_cash_spent']),
                "avg_price": float(results['best_ask']['avg_price']),
                "executed": float(results['best_ask']['executed']),
                "remaining": float(results['best_ask']['remaining'])
            },
            "twap": {
                "total_cash_spent": float(results['twap']['total_cash_spent']),
                "avg_price": float(results['twap']['avg_price']),
                "executed": float(results['twap']['executed']),
                "remaining": float(results['twap']['remaining'])
            },
            "vwap": {
                "total_cash_spent": float(results['vwap']['total_cash_spent']),
                "avg_price": float(results['vwap']['avg_price']),
                "executed": float(results['vwap']['executed']),
                "remaining": float(results['vwap']['remaining'])
            }
        }
    }

    print(json.dumps(output, indent=2))

    with open('results.json', 'w') as f:
        json.dump(output, f, indent=2)

    return output

if __name__ == '__main__':
    results = parameter_search(None, order_size=5000)
