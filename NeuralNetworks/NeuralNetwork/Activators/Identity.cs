using System;
using System.Collections.Generic;
using System.Text;
using NeuralNetwork.Common.Activators;
namespace NeuralNetwork.Activators
{
    public  class Identity : IActivator
    {
        public Identity()
        {

        }
        public Func<double, double> Apply => d => d;
        public Func<double, double> ApplyDerivative => d => 1;

        public ActivatorType Type => ActivatorType.Identity;
    }
}
