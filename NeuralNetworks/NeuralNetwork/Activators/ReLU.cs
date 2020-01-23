using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Common.Activators
{
    class ReLU : IActivator
    {
        public ReLU()
        {

        }
        public Func<double, double> Apply => d => Math.Max(0,d);
        public Func<double, double> ApplyDerivative => d => Convert.ToDouble(d>0);

        public ActivatorType Type => ActivatorType.ReLU;
    }
}
