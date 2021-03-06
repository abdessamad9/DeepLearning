﻿using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;
namespace NeuralNetwork.Activators
{
    public class Sigmoid : IActivator
    {
        public Sigmoid()
        {

        }
        public Func<double, double> Apply => d => 1.0/(1.0+Math.Exp(-d));
        public Func<double, double> ApplyDerivative => d => 1.0 / (1.0 + Math.Exp(-d)) * (1.0- 1.0 / (1.0 + Math.Exp(-d))) ;

        public ActivatorType Type => ActivatorType.Sigmoid;
    }
    
}
