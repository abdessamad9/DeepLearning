using System;
using System.Collections.Generic;
using System.Text;

namespace NeuralNetwork.Common.Activators
{
    class LeakyReLU : IActivator
        {
        public LeakyReLU()
        {

        }
       //A modifier 
        public Func<double, double> Apply => d => Math.Max(0, d);
        public Func<double, double> ApplyDerivative => d => Convert.ToDouble(d > 0);

        public ActivatorType Type => ActivatorType.LeakyReLU;
    }
    
}
