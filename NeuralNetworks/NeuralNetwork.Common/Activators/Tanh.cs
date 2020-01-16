using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Common.Activators
{
    class Tanh : IActivator
    {
        public Tanh()
        {

        }
        public Func<double, double> Apply => d => (exp(d) - exp(-d))/(exp(d) + exp(-d));
        public Func<double, double> ApplyDerivative => d => 1-((exp(d) - exp(-d)) / (exp(d) + exp(-d))) * ((exp(d) - exp(-d)) / (exp(d) + exp(-d)));

        public ActivatorType Type => ActivatorType.Tanh;
    }
}
