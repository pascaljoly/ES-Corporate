# Energy Score Tool - Demo Scripts

## Demo Options

### Option 1: Interactive Demo (Recommended for Presentations)
```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python interactive_demo.py
```
**Perfect for live presentations** - pauses between each step, allows you to explain what's happening.

### Option 2: Complete Demo (2-3 minutes)
```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python end_to_end_demo.py
```
Runs automatically without pauses.

### Option 3: Quick Demo (30 seconds)
```bash
cd /path/to/EStool
source energy-compare/venv/bin/activate
python quick_demo.py
```
Fast overview for brief meetings.

## Interactive Demo Flow

The interactive demo walks through 5 clear steps:

### **Step 1: Individual Model Scoring**
- Score each model one by one
- Pause between each model to explain results
- Show energy, CO2, throughput, and model details

### **Step 2: Model Comparison & Ranking**
- Compare text generation models
- Compare computer vision models
- Show rankings with star ratings
- Pause to explain the comparison process

### **Step 3: Custom Scoring Weights**
- Demonstrate energy-focused comparison
- Demonstrate performance-focused comparison
- Show how rankings change with different priorities
- Explain business use cases for each approach

### **Step 4: Configuration System**
- Show current configuration settings
- Demonstrate environment variable overrides
- Explain how to customize without code changes

### **Step 5: Business Value**
- Summarize key business benefits
- Show real-world use cases
- Highlight technical advantages

## What the Demo Shows

### 1. **Model Energy Scoring**
- Scores individual ML models for energy consumption
- Shows energy (kWh), CO2 emissions (kg), throughput (samples/sec)
- Displays model size and architecture information

### 2. **Model Comparison & Ranking**
- Compares multiple models side-by-side
- Ranks models using 1-5 star rating system
- Shows clear winners and performance trade-offs

### 3. **Custom Scoring Weights**
- Demonstrates different prioritization strategies
- Energy-focused vs Performance-focused comparisons
- Shows how rankings change based on business priorities

### 4. **Configuration System**
- Shows centralized configuration management
- Demonstrates environment variable overrides
- Highlights ease of customization without code changes

### 5. **Business Value**
- Cost optimization through energy-efficient model selection
- Sustainability and CO2 emission tracking
- Real-world use cases and technical advantages

## Expected Output

The demo will show:
- ✅ Individual model scoring results
- ✅ Model comparison rankings with star ratings
- ✅ Custom weight demonstrations
- ✅ Configuration system capabilities
- ✅ Business value proposition

## Demo Duration

Approximately 2-3 minutes for the complete demo.

## Key Messages for Management

1. **Cost Savings**: Choose energy-efficient models to reduce cloud costs
2. **Sustainability**: Track and minimize CO2 emissions from ML workloads
3. **Flexibility**: Easy customization without code changes
4. **Standardization**: Consistent 1-5 star rating system across all models
5. **Production Ready**: Complete end-to-end solution for model evaluation
