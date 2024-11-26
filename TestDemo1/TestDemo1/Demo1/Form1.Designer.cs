namespace Demo1
{
    partial class Form1
    {
        /// <summary>
        ///  Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        ///  Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        ///  Required method for Designer support - do not modify
        ///  the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            label1 = new Label();
            textBox1 = new TextBox();
            buttonTrial1 = new ButtonTrial();
            SuspendLayout();
            // 
            // label1
            // 
            label1.AutoSize = true;
            label1.BackColor = Color.PeachPuff;
            label1.Location = new Point(71, 156);
            label1.Name = "label1";
            label1.Size = new Size(92, 15);
            label1.TabIndex = 0;
            label1.Text = "Input Your Hash";
            // 
            // textBox1
            // 
            textBox1.Location = new Point(188, 153);
            textBox1.Name = "textBox1";
            textBox1.Size = new Size(494, 23);
            textBox1.TabIndex = 1;
            // 
            // buttonTrial1
            // 
            buttonTrial1.BackColor = Color.LightSlateGray;
            buttonTrial1.FlatAppearance.BorderSize = 0;
            buttonTrial1.FlatStyle = FlatStyle.Flat;
            buttonTrial1.ForeColor = Color.White;
            buttonTrial1.Location = new Point(326, 261);
            buttonTrial1.Name = "buttonTrial1";
            buttonTrial1.Size = new Size(142, 40);
            buttonTrial1.TabIndex = 3;
            buttonTrial1.Text = "Submit";
            buttonTrial1.UseVisualStyleBackColor = false;
            buttonTrial1.Click += buttonTrial1_Click;
            // 
            // Form1
            // 
            AutoScaleDimensions = new SizeF(7F, 15F);
            AutoScaleMode = AutoScaleMode.Font;
            BackColor = SystemColors.ScrollBar;
            ClientSize = new Size(800, 450);
            Controls.Add(buttonTrial1);
            Controls.Add(textBox1);
            Controls.Add(label1);
            Name = "Form1";
            Text = "Form1";
            ResumeLayout(false);
            PerformLayout();
        }

        #endregion

        private Label label1;
        private TextBox textBox1;
        private ButtonTrial buttonTrial1;
    }
}
