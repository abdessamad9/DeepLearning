using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;
namespace NeuralNetwork.Activators
{
    class Tanh : IActivator
    {
        public Tanh()
        {

        }
        public Func<double, double> Apply => d => (Math.Exp(d) - Math.Exp(-d))/(Math.Exp(d) + Math.Exp(-d));
        public Func<double, double> ApplyDerivative => d => 1-((Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d))) * ((Math.Exp(d) - Math.Exp(-d)) / (Math.Exp(d) + Math.Exp(-d)));

        public ActivatorType Type => ActivatorType.Tanh;
    }
}
