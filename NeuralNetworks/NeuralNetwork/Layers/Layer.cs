using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using NeuralNetwork.Common.Activators;
using NeuralNetwork.Common.GradientAdjustmentParameters;
using NeuralNetwork.Common.GradientAdjustmentsParameters;
using NeuralNetwork.Common.Layers;


namespace NeuralNetwork.Layers
{
    public interface Layer : ILayer
    {

        //Type Qui permet d'avoir le type du Layer
        LayerType Type { get; }
      
    }
}
