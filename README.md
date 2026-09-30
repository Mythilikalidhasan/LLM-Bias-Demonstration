# LLM Bias Demonstration

An interactive **AI Bias Demonstration** web application built using Python and Streamlit. This project demonstrates how biased training data can influence the patterns learned by an AI model and affect its responses.

## Technologies Used

- **Python** – Core programming language
- **Streamlit** – Interactive web application development
- **Scikit-learn** – Machine learning, TF-IDF and Logistic Regression
- **Pandas** – Data handling and processing
- **HTML & CSS** – Custom UI design and styling
- **Git & GitHub** – Version control and project hosting
- **Streamlit Community Cloud** – Deployment

##  Working Flow

Training Data
      ↓
TF-IDF Vectorization
      ↓
Machine Learning Model
(Logistic Regression)
      ↓
User Question
      ↓
AI Response
      ↓
Bias Detection
      ↓
Balance Training Data
      ↓
Compare Results

##  How It Works

The application demonstrates the relationship between training data and AI responses.

1. The user views the training data.
2. The model is trained using the available data.
3. The user enters a question.
4. The system generates a response based on learned patterns.
5. Potential bias is displayed using a bias indicator.
6. The user can balance the training data.
7. The model is retrained and the results are compared.

### Architecture

<img width="3150" height="7107" alt="diagram" src="https://github.com/user-attachments/assets/28874b1c-f169-433d-8b00-5aef37139e4a" />


## Future Enhancements

- Integrate a real Large Language Model (LLM)
- Add more types of bias such as cultural, language, and racial bias
- Use larger and more diverse datasets
- Add advanced bias detection techniques
- Provide interactive charts and analytics
- Add model performance metrics
- Allow users to upload their own datasets
- Improve the model with human feedback
- Add multilingual support
- Deploy the application with a custom domain

##  Conclusion

This project provides a simple and interactive way to understand how training data can influence AI behavior. It highlights the importance of **diverse, balanced, and carefully reviewed data** when developing AI systems.
