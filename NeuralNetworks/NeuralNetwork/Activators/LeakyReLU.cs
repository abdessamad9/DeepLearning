using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;
namespace NeuralNetwork.Activators
{
    class LeakyReLU : IActivator
    {
        double alpha;
        public LeakyReLU()
        {
            alpha = 0.01;
        }
        public Func<double, double> Apply => d => Convert.ToDouble(d < 0) * alpha * d + Convert.ToDouble(d > 0) * d;
        public Func<double, double> ApplyDerivative => d => alpha * Convert.ToDouble(d < 0) + Convert.ToDouble(d > 0);

        public ActivatorType Type => ActivatorType.LeakyReLU;
    }

}
