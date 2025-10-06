from flask import Flask, render_template, request, jsonify
import pandas as pd
from recommend import top_n_for_user, explain_reasons

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["POST"])
def recommend_route():
    user_id = int(request.form.get("user_id", "1"))
    n = int(request.form.get("n", "10"))
    try:
        recs = top_n_for_user(user_id, n)
        return render_template(
            "index.html",
            user_id=user_id,
            n=n,
            table=recs.to_html(classes="table table-striped", index=False, justify="left")
        )
    except Exception as e:
        return render_template("index.html", error=str(e)), 400

# simple JSON APIs (useful for screenshots in the report)
@app.route("/api/recommend")
def api_recommend():
    user_id = int(request.args.get("user_id", "1"))
    n = int(request.args.get("n", "10"))
    recs = top_n_for_user(user_id, n)
    return jsonify(recs.to_dict(orient="records"))

@app.route("/api/explain")
def api_explain():
    user_id = int(request.args.get("user_id", "1"))
    movie_id = int(request.args.get("movie_id"))
    reasons = explain_reasons(user_id, movie_id, k=3)
    return jsonify([{"title": t, "your_rating": r} for (t, r) in reasons])

if __name__ == "__main__":
    app.run(debug=True)
