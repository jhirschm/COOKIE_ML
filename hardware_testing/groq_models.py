from groqflow import groqit
import torch

gmodel = groqit(pytorch_model, inputs, groqview=True, rebuild="always", build_name="model "+str(count))
  gmodel.groqview()
   # Get performance estimates in terms of latency and throughput
  estimate = gmodel.estimate_performance()
  print("Your build's estimated performance is:")
  print(f"{estimate.latency:.7f} {estimate.latency_units}")
  print(f"{estimate.throughput:.1f} {estimate.throughput_units}")
  print("Example estimate_performance.py finished")
