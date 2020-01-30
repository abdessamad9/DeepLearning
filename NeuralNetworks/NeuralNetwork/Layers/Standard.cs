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
    public class Standard : ILayer
    {
        //
        public Standard( int batchSize, IActivator activator, double[] bias, double[,] weights, IGradientAdjustmentParameters gradientAdjustmentParameters, double indiceLayer)
        {

            BatchSize = batchSize;
            LayerSize = bias.Length;
            InputSize = weights.GetLength(0);
            Activator =activator;
            Bias = Matrix<double>.Build.Dense(bias.Length, BatchSize,0);
            for(int i= 0; i < BatchSize; i++)
            {
                Bias.SetColumn(i, bias);
            }
            B = Matrix<double>.Build.Dense(LayerSize, BatchSize,0);
            Weights = DenseMatrix.OfArray(weights);
            WeightedError = Matrix<double>.Build.Dense(Weights.RowCount,B.ColumnCount,0);
            Activation = Matrix<double>.Build.Dense(LayerSize, BatchSize,0);
            GradientAdjustmentParameters = gradientAdjustmentParameters;
            VelocityWeights = Matrix<double>.Build.Dense(Weights.RowCount, Weights.ColumnCount, 0);
            VelocityBias = Matrix<double>.Build.Dense(Bias.RowCount, Bias.ColumnCount, 0);

            SWeights = Matrix<double>.Build.Dense(Weights.RowCount, Weights.ColumnCount, 0);
            SBias = Matrix<double>.Build.Dense(Bias.RowCount, Bias.ColumnCount, 0);
            RWeights = Matrix<double>.Build.Dense(Weights.RowCount, Weights.ColumnCount, 0);
            RBias = Matrix<double>.Build.Dense(Bias.RowCount, Bias.ColumnCount, 0);
            SPWeights = Matrix<double>.Build.Dense(Weights.RowCount, Weights.ColumnCount, 0);
            SPBias = Matrix<double>.Build.Dense(Bias.RowCount, Bias.ColumnCount, 0);
            RPWeights = Matrix<double>.Build.Dense(Weights.RowCount, Weights.ColumnCount, 0);
            RPBias = Matrix<double>.Build.Dense(Bias.RowCount, Bias.ColumnCount, 0);

            IndiceLayer = indiceLayer;
        }
        double indiceLayer;
        public double IndiceLayer
        {
            get
            {
                return indiceLayer;
            }
            set
            {
                this.indiceLayer = value;
            }
        }
        Matrix<double> rPBias;
        public Matrix<double> RPBias
        {
            get
            {
                return rPBias;

            }
            set
            {
                this.rPBias = value;
            }
        }
        Matrix<double> rBias;
        public Matrix<double> RBias
        {
            get
            {
                return rBias;

            }
            set
            {
                this.rBias = value;
            }
        }

        Matrix<double> sPBias;
        public Matrix<double> SPBias
        {
            get
            {
                return sPBias;

            }
            set
            {
                this.sPBias = value;
            }
        }
        Matrix<double> sBias;
        public Matrix<double> SBias
        {
            get
            {
                return sBias;

            }
            set
            {
                this.sBias = value;
            }
        }



        Matrix<double> rPWeights;
        public Matrix<double> RPWeights
        {
            get
            {
                return rPWeights;

            }
            set
            {
                this.rPWeights = value;
            }
        }
        Matrix<double> rWeights;
        public Matrix<double> RWeights
        {
            get
            {
                return rWeights;

            }
            set
            {
                this.rWeights = value;
            }
        }

        Matrix<double> sPWeights;
        public Matrix<double> SPWeights
        {
            get
            {
                return sPWeights;

            }
            set
            {
                this.sPWeights = value;
            }
        }
        Matrix<double> sWeights;
        public Matrix<double> SWeights
        {
            get
            {
                return sWeights;

            }
            set
            {
                this.sWeights = value;
            }
        }

        Matrix<double> velocityWeights;
        public Matrix<double> VelocityWeights
        {
            get
            {
                return velocityWeights;

            }
            set
            {
                this.velocityWeights = value;
            }
        }

        Matrix<double> velocityBias;
        public Matrix<double> VelocityBias
        {
            get
            {
                return velocityBias;

            }
            set
            {
                this.velocityBias = value;
            }
        }

        Matrix<double> lastActivation;
        public Matrix<double> LastActivation
        {
            get
            {
                return lastActivation;
            }
            set
            {
                lastActivation = value;
            }
        }
        Matrix<double> b;
        public Matrix<double> B
        {
            get
            {
                return b;
            }
            set
            {
                b = value;
            }
        }
        Matrix<double> weights;
        public Matrix<double> Weights
        {
            get
            {
                return weights;
            }
            set
            {
                weights = value;
            }
        }
        IGradientAdjustmentParameters gradientAdjustmentParameters;
        public IGradientAdjustmentParameters GradientAdjustmentParameters
        {
            get
            {
                return gradientAdjustmentParameters;
            }
            set
            {
                this.gradientAdjustmentParameters =  value;
            }
        }

        int layerSize;
        public int LayerSize
        {
            get
            {
                return layerSize;
            }
            set
            {
                this.layerSize = value;
            }
        }
        int inputSize;
        public int InputSize
        {
            get
            {
                return inputSize;
            }
            set
            {
                this.inputSize = value;
            }
        }
        IActivator activator;
        public IActivator Activator
        {
            get
            {
                return activator;
            }
            set
            {
                this.activator = value;
            }
        }

        Matrix<double> bias;
        public Matrix<double> Bias
        {
            get
            {
                return bias;
            }
            set
            {
                this.bias = value;
            }
        }

        int batchSize;
        public int BatchSize
        {
            get
            {
                return batchSize;
            }
            set
            {
                this.batchSize = value;
            }
        }
        Matrix<double> zeta;
        public Matrix<double> Zeta
        {
            get
            {
                return zeta;
            }
            set
            {
                this.zeta = value;
            }
        }
        Matrix<double> activation;
        public Matrix<double> Activation
        {
            get
            {
                return activation;
            }
            set
            {
                this.activation = value;
            }
        }

        Matrix<double> weightedError;
        public Matrix<double> WeightedError
        {
            get
            {
                return weightedError;
            }
            set
            {
                this.weightedError = value;
            }
        }
        
        public bool Equals(ILayer other)
        {
            return  LayerSize == other.LayerSize && InputSize == other.InputSize && BatchSize == other.BatchSize;
        }

        public void Propagate(Matrix<double> input)
        {
                                                               
            LastActivation = input;
            Zeta = weights.Transpose() * input + Bias;
            Zeta.Map(Activator.Apply, Activation);
        }

        public void UpdateParameters()
        {
            Computation();
            UpdateBias(VelocityBias, 1.0);
            UpdateWeights(VelocityWeights, 1.0);
        }

        public double Computation()
        {
            //Vector<double> res;
            var gradW = 1.0 / BatchSize * LastActivation * B.Transpose();
            var gradB = 1.0 / BatchSize * B * Vector<double>.Build.Dense(BatchSize, 1);
            Matrix<double> gradBvect = Matrix<double>.Build.Dense(SBias.RowCount, SBias.ColumnCount, 0);
            for (int i = 0; i < BatchSize; i++)
            {
                gradBvect.SetColumn(i, gradB);
            }
            double coefficient = 1.0;
            switch (GradientAdjustmentParameters.Type)
            {
                case GradientAdjustmentType.FixedLearningRate:
                    coefficient = ((FixedLearningRateParameters)(GradientAdjustmentParameters)).LearningRate;
                    VelocityWeights.Multiply(-coefficient, VelocityWeights);    
                    gradBvect.Multiply(-coefficient, gradBvect);
                    gradBvect.CopyTo(VelocityBias);
                    
                    break;

                case GradientAdjustmentType.Adam:
                    coefficient = ((AdamParameters)(GradientAdjustmentParameters)).StepSize;
                    SWeights.Multiply(((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay, SWeights);
                    SWeights.Add((1.0 - ((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay) * gradW, SWeights);
                    RWeights.Multiply(((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay);
                    var g2 = gradW.PointwiseMultiply(gradW);
                    RWeights.Add((1.0 - ((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay) * g2, RWeights);
                    SWeights.CopyTo(SPWeights);
                    SPWeights.Multiply(1.0 / (1.0 - Math.Pow(((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay, IndiceLayer)), SPWeights);
                    RWeights.CopyTo(RPWeights);
                    RPWeights.Multiply(1.0 / (1.0 - Math.Pow(((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay, IndiceLayer)), RPWeights);
                    SPWeights.Multiply(-coefficient, SPWeights);
                    var sqrtrp = RPWeights.Clone();
                    RPWeights.Map(Math.Sqrt, sqrtrp);
                    sqrtrp.Add(((AdamParameters)(GradientAdjustmentParameters)).DenominatorFactor, sqrtrp);
                    VelocityWeights = SPWeights.PointwiseDivide(sqrtrp);
                    

                    SBias.Multiply(((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay, SBias);
                    SBias.Add((1.0 - ((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay) * gradBvect, SBias);
                    RBias.Multiply(((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay, RBias);
                    var gB2 = gradBvect.PointwiseMultiply(gradBvect);
                    RBias.Add((1.0 - ((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay) * gB2, RBias);
                    SBias.CopyTo(SPBias);
                    SPBias.Multiply(1.0 / (1.0 - Math.Pow(((AdamParameters)(GradientAdjustmentParameters)).FirstMomentDecay, IndiceLayer)), SPBias);
                    RBias.CopyTo(RPBias);
                    RPBias.Multiply(1.0 / (1.0 - Math.Pow(((AdamParameters)(GradientAdjustmentParameters)).SecondMomentDecay, IndiceLayer)), RPBias);
                    SPBias.Multiply(-coefficient, SPBias);
                    var sqrtrpB = RPBias.Clone();
                    RPBias.Map(Math.Sqrt, sqrtrpB);
                    sqrtrpB.Add(((AdamParameters)(GradientAdjustmentParameters)).DenominatorFactor, sqrtrpB);
                    VelocityBias = SPBias.PointwiseDivide(sqrtrpB);
                    
                    break;

                case GradientAdjustmentType.Momentum:
                    coefficient = ((MomentumParameters)(GradientAdjustmentParameters)).LearningRate;
                    velocityBias.Multiply(((MomentumParameters)(GradientAdjustmentParameters)).Momentum);
                    for (int i = 0; i < BatchSize; i++)
                    {
                        velocityBias.SetColumn(i, velocityBias.Column(i) - coefficient * gradB);
                    }
                    
                    velocityWeights.Multiply(((MomentumParameters)(GradientAdjustmentParameters)).Momentum, velocityWeights);
                    velocityWeights.Add(-coefficient * gradW, velocityWeights);
                    
                    break;

                default:
                    throw new InvalidOperationException("Unknown gradient accelerator parameter");
            }
            return coefficient;
        }

        public void UpdateWeights(Matrix<double> velocity, double coefficient)
        {
            Weights.Multiply(coefficient, Weights);
            Weights.Add(velocity, Weights);
        }
        public void UpdateBias(Matrix<double> velocity, double coefficient)
        {
            Bias.Multiply(coefficient, Bias);
            Bias.Add(velocity, Bias);
        }
        public void BackPropagate(Matrix<double> upstreamWeightedErrors)
        {
            Matrix<double> ZetaPrime = Matrix<double>.Build.Dense(LayerSize,BatchSize,0);
            zeta.Map(Activator.ApplyDerivative,ZetaPrime);
            ZetaPrime.PointwiseMultiply(upstreamWeightedErrors,B);
            WeightedError = Weights * B;

        }

       
    }
}
