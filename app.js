// SmartCart Dashboard & Machine Learning Inference Engine

document.addEventListener("DOMContentLoaded", () => {
    // 1. Tab Navigation Handling
    const navButtons = document.querySelectorAll(".nav-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    const pageTitle = document.getElementById("page-title");
    const pageDescription = document.getElementById("page-description");

    navButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            const targetTab = btn.getAttribute("data-tab");
            
            navButtons.forEach(b => b.classList.remove("active"));
            tabContents.forEach(tc => tc.classList.remove("active"));
            
            btn.classList.add("active");
            document.getElementById(targetTab).classList.add("active");

            if (targetTab === "dashboard-tab") {
                pageTitle.innerText = "📊 Customer Intelligence Dashboard";
                pageDescription.innerText = "Explore customer segments, purchasing behavior, and model evaluations dynamically in real-time.";
            } else {
                pageTitle.innerText = "🔮 Predict Segment & Churn";
                pageDescription.innerText = "Execute live machine learning predictions and retention strategies on unique customer profiles.";
            }
        });
    });

    // 2. Initialize Dashboard Stats & Summaries
    initDashboard();

    // 3. Form Submission & Machine Learning Predictions
    const predForm = document.getElementById("prediction-form");
    predForm.addEventListener("submit", handleFormSubmit);
});

// Helper dictionaries matching Streamlit python app
const SEGMENT_NAMES = {
    0: "Budget Families",
    1: "Moderate Spenders",
    2: "Premium Customers",
    3: "Budget Singles"
};

const SEGMENT_DESCRIPTIONS = {
    0: "Lower income, budget-conscious spenders who typically live with a partner.",
    1: "Middle-income earners with moderate spending. Typically older families.",
    2: "High-income, high-spending premium customers. They have the fewest children.",
    3: "Lower income, budget-conscious spenders who live alone."
};

const SEGMENT_COLORS = {
    0: "#38bdf8", // Sky blue
    1: "#fb923c", // Orange
    2: "#d8b4fe", // Lavender/purple
    3: "#34d399"  // Emerald green
};

const SEGMENT_GLOWS = {
    0: "rgba(56, 189, 248, 0.2)",
    1: "rgba(251, 146, 60, 0.2)",
    2: "rgba(216, 180, 254, 0.3)",
    3: "rgba(52, 211, 153, 0.2)"
};

const SEGMENT_ICONS = {
    0: "🛒",
    1: "👨‍👩‍👧‍👦",
    2: "💎",
    3: "👤"
};

let scatterChartInstance = null;
let spendingChartInstance = null;
let channelsChartInstance = null;

function initDashboard() {
    // Set cluster sizes
    document.getElementById("size-seg-0").innerText = DATASET_SUMMARY.cluster_sizes[0].toLocaleString();
    document.getElementById("size-seg-1").innerText = DATASET_SUMMARY.cluster_sizes[1].toLocaleString();
    document.getElementById("size-seg-2").innerText = DATASET_SUMMARY.cluster_sizes[2].toLocaleString();
    document.getElementById("size-seg-3").innerText = DATASET_SUMMARY.cluster_sizes[3].toLocaleString();

    // Load table content
    const tableBody = document.getElementById("summary-table-body");
    tableBody.innerHTML = "";
    
    // Sort keys and iterate
    for (let c = 0; c < 4; c++) {
        const stats = DATASET_SUMMARY.cluster_summary[c];
        const tr = document.createElement("tr");
        
        // Highlight active row border
        tr.style.borderLeft = `3px solid ${SEGMENT_COLORS[c]}`;
        
        tr.innerHTML = `
            <td style="font-weight: 600; color: #fff;">${SEGMENT_ICONS[c]} ${SEGMENT_NAMES[c]}</td>
            <td>$${stats.Income.toLocaleString()}</td>
            <td>${stats.Recency} days</td>
            <td>${stats.NumDealsPurchases}</td>
            <td>${stats.NumWebPurchases}</td>
            <td>${stats.NumCatalogPurchases}</td>
            <td>${stats.NumStorePurchases}</td>
            <td>${stats.NumWebVisitsMonth}</td>
            <td>${stats.Age} yrs</td>
            <td>${stats.Customer_Tenure_Days.toFixed(0)}</td>
            <td style="color: ${SEGMENT_COLORS[c]}; font-weight: 600;">$${stats.Total_Spending.toLocaleString()}</td>
            <td>${stats.Total_Children}</td>
        `;
        tableBody.appendChild(tr);
    }

    // Render Charts
    renderCharts();
}

function renderCharts() {
    // 1. Income vs Spending Scatter Chart
    const scatterCtx = document.getElementById('scatterChart').getContext('2d');
    
    // Group scatter data by cluster
    const clusterScatterData = { 0: [], 1: [], 2: [], 3: [] };
    DATASET_SUMMARY.scatter_data.forEach(item => {
        clusterScatterData[item.cluster].push({ x: item.spending, y: item.income });
    });

    const scatterDatasets = Object.keys(clusterScatterData).map(clusterId => {
        const cId = parseInt(clusterId);
        return {
            label: SEGMENT_NAMES[cId],
            data: clusterScatterData[cId],
            backgroundColor: SEGMENT_COLORS[cId],
            borderColor: SEGMENT_COLORS[cId],
            borderWidth: 1,
            pointRadius: 4,
            pointHoverRadius: 7,
            alpha: 0.6
        };
    });

    if (scatterChartInstance) scatterChartInstance.destroy();
    scatterChartInstance = new Chart(scatterCtx, {
        type: 'scatter',
        data: { datasets: scatterDatasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#e5e7eb', font: { family: 'Inter', weight: 500 } }
                },
                tooltip: {
                    backgroundColor: '#1f2937',
                    titleColor: '#fff',
                    bodyColor: '#e5e7eb',
                    borderColor: 'rgba(255,255,255,0.1)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            return `${context.dataset.label} | Spend: $${context.raw.x.toLocaleString()} | Income: $${context.raw.y.toLocaleString()}`;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Total Spending ($)', color: '#9ca3af', font: { weight: 600 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                },
                y: {
                    title: { display: true, text: 'Income ($)', color: '#9ca3af', font: { weight: 600 } },
                    grid: { color: 'rgba(255,255,255,0.05)' },
                    ticks: { color: '#9ca3af' }
                }
            }
        }
    });

    // 2. Average Total Spending Bar Chart
    const spendingCtx = document.getElementById('spendingBarChart').getContext('2d');
    const spendingLabels = Object.values(SEGMENT_NAMES);
    const spendingValues = Object.keys(DATASET_SUMMARY.spending_by_cluster).map(c => DATASET_SUMMARY.spending_by_cluster[c]);
    const spendingColors = Object.values(SEGMENT_COLORS);

    if (spendingChartInstance) spendingChartInstance.destroy();
    spendingChartInstance = new Chart(spendingCtx, {
        type: 'bar',
        data: {
            labels: spendingLabels,
            datasets: [{
                data: spendingValues,
                backgroundColor: spendingColors,
                borderRadius: 8,
                borderWidth: 0,
                barThickness: 35
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: '#1f2937',
                    callbacks: {
                        label: (c) => ` Avg Spend: $${c.raw.toLocaleString()}`
                    }
                }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 11 } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
            }
        }
    });

    // 3. Purchasing Channels grouped bar chart
    const channelsCtx = document.getElementById('channelsBarChart').getContext('2d');
    const webData = [];
    const storeData = [];
    const catalogData = [];

    for (let c = 0; c < 4; c++) {
        webData.push(DATASET_SUMMARY.channel_data_by_cluster[c].web);
        storeData.push(DATASET_SUMMARY.channel_data_by_cluster[c].store);
        catalogData.push(DATASET_SUMMARY.channel_data_by_cluster[c].catalog);
    }

    if (channelsChartInstance) channelsChartInstance.destroy();
    channelsChartInstance = new Chart(channelsCtx, {
        type: 'bar',
        data: {
            labels: spendingLabels,
            datasets: [
                { label: 'Web', data: webData, backgroundColor: '#3b82f6', borderRadius: 6 },
                { label: 'Store', data: storeData, backgroundColor: '#10b981', borderRadius: 6 },
                { label: 'Catalog', data: catalogData, backgroundColor: '#a855f7', borderRadius: 6 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'top',
                    labels: { color: '#e5e7eb', boxWidth: 12, boxHeight: 12 }
                },
                tooltip: { backgroundColor: '#1f2937' }
            },
            scales: {
                x: { grid: { display: false }, ticks: { color: '#9ca3af', font: { size: 10 } } },
                y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9ca3af' } }
            }
        }
    });
}

// Client-side inference math formulas
function predictSegment(input) {
    // 1. One hot encode Education and Living_With
    const ohe = {
        Education_Graduate: input.Education === "Graduate" ? 1.0 : 0.0,
        Education_Postgraduate: input.Education === "Postgraduate" ? 1.0 : 0.0,
        Education_Undergraduate: input.Education === "Undergraduate" ? 1.0 : 0.0,
        Living_With_Alone: input.Living_With === "Alone" ? 1.0 : 0.0,
        Living_With_Partner: input.Living_With === "Partner" ? 1.0 : 0.0
    };

    // Merge input variables
    const raw_features = { ...input, ...ohe };

    // 2. Scale features using scaler parameters
    const feature_names = MODEL_PARAMETERS.scaler.feature_names;
    const scaled_features = [];
    for (let i = 0; i < feature_names.length; i++) {
        const col = feature_names[i];
        const mean = MODEL_PARAMETERS.scaler.mean[i];
        const scale = MODEL_PARAMETERS.scaler.scale[i];
        const val = raw_features[col] !== undefined ? raw_features[col] : 0.0;
        scaled_features.push((val - mean) / scale);
    }

    // 3. Project scaled vector using PCA components: (scaled - pca_mean) @ components.T
    const pca_mean = MODEL_PARAMETERS.pca.mean;
    const pca_components = MODEL_PARAMETERS.pca.components;
    
    // Subtract PCA mean
    const diff = [];
    for (let i = 0; i < scaled_features.length; i++) {
        diff.push(scaled_features[i] - pca_mean[i]);
    }

    // Dot product with PCA eigenvectors (components)
    const pca_transformed = [];
    for (let c = 0; c < pca_components.length; c++) {
        let dot = 0.0;
        for (let i = 0; i < scaled_features.length; i++) {
            dot += diff[i] * pca_components[c][i];
        }
        pca_transformed.push(dot);
    }

    // 4. KMeans Cluster Assignment: closest Euclidean distance in PCA space
    const centers = MODEL_PARAMETERS.kmeans.cluster_centers;
    let min_dist = Infinity;
    let predicted_cluster = 0;
    
    for (let k = 0; k < centers.length; k++) {
        let dist = 0.0;
        for (let d = 0; d < 3; d++) {
            dist += Math.pow(centers[k][d] - pca_transformed[d], 2);
        }
        if (dist < min_dist) {
            min_dist = dist;
            predicted_cluster = k;
        }
    }

    return predicted_cluster;
}

function predictChurn(input) {
    // Random Forest recursive evaluation
    const trees = MODEL_PARAMETERS.churn.trees;
    const features = MODEL_PARAMETERS.churn.features;
    
    function evaluateTree(node) {
        if (node.leaf) {
            return node.value[1]; // probability of class 1 (churn)
        }
        const feat_name = features[node.feature_idx];
        const val = input[feat_name] !== undefined ? input[feat_name] : 0.0;
        if (val <= node.threshold) {
            return evaluateTree(node.left);
        } else {
            return evaluateTree(node.right);
        }
    }

    let total_prob = 0.0;
    for (let i = 0; i < trees.length; i++) {
        total_prob += evaluateTree(trees[i]);
    }
    return total_prob / trees.length;
}

function handleFormSubmit(e) {
    e.preventDefault();

    // Read form values
    const age = parseInt(document.getElementById("age").value);
    const income = parseFloat(document.getElementById("income").value);
    const customer_tenure = parseFloat(document.getElementById("customer_tenure").value);
    const total_children = parseInt(document.getElementById("total_children").value);
    const education = document.getElementById("education").value;
    const living_with = document.getElementById("living_with").value;

    const total_spending = parseFloat(document.getElementById("total_spending").value);
    const num_deals = parseInt(document.getElementById("num_deals").value);
    const num_web = parseInt(document.getElementById("num_web").value);
    const num_catalog = parseInt(document.getElementById("num_catalog").value);
    const num_store = parseInt(document.getElementById("num_store").value);

    const recency = parseInt(document.getElementById("recency").value);
    const num_web_visits = parseInt(document.getElementById("num_web_visits").value);
    const complain = parseInt(document.getElementById("complain").value);
    const response = parseInt(document.getElementById("response").value);

    // Combine parameters
    const input_data = {
        Age: age,
        Income: income,
        Customer_Tenure_Days: customer_tenure,
        Total_Children: total_children,
        Education: education,
        Living_With: living_with,
        Total_Spending: total_spending,
        NumDealsPurchases: num_deals,
        NumWebPurchases: num_web,
        NumCatalogPurchases: num_catalog,
        NumStorePurchases: num_store,
        Recency: recency,
        NumWebVisitsMonth: num_web_visits,
        Complain: complain,
        Response: response
    };

    // 1. Run cluster segment prediction
    const prediction = predictSegment(input_data);

    // 2. Run Churn prediction
    const churn_prob = predictChurn(input_data);
    const churn_risk = churn_prob > 0.50 ? "HIGH" : "LOW";

    // 3. Render prediction details
    document.getElementById("res-segment-icon").innerText = SEGMENT_ICONS[prediction];
    document.getElementById("res-segment-title").innerText = SEGMENT_NAMES[prediction];
    
    // Set custom visual borders & shadows on the segment result card based on predicted group
    const segCard = document.getElementById("result-segment-card");
    segCard.style.borderColor = SEGMENT_COLORS[prediction];
    segCard.querySelector(".card-glow").style.background = `radial-gradient(circle, ${SEGMENT_GLOWS[prediction]} 0%, transparent 75%)`;
    
    const tag = document.getElementById("res-segment-tagline");
    tag.style.color = SEGMENT_COLORS[prediction];

    // Configure Segment Specific Insights & Strategies
    let tagText = "";
    let businessInsight = "";
    let recStrategy = "";
    let recCampaign = "";
    let recOffer = "";
    let recPriority = "";
    let recActions = [];

    if (prediction === 2) {
        tagText = "💎 PREMIUM SPENDER | HIGH VALUE";
        businessInsight = "These customers generate high revenue and are strongly engaged with the brand.";
        recStrategy = "Target with premium offers and exclusive deals";
        recCampaign = "VIP Email Marketing & Exclusive Previews";
        recOffer = "Early Access to Luxury Products & Free Shipping";
        recPriority = "High";
        recActions = ["Promote luxury products", "Give early access VIP deals", "Assign high-touch customer support representative"];
    } else if (prediction === 0) {
        tagText = "🛒 BUDGET FAMILY | PRICE SENSITIVE";
        businessInsight = "These customers are highly price-sensitive and respond well to discounts and family bundles.";
        recStrategy = "Provide discounts and bundle offers";
        recCampaign = "Discount/Promo Newsletters & Coupon booklets";
        recOffer = "20% Discount / Buy 1 Get 1 Free";
        recPriority = "Medium";
        recActions = ["Send bundle offers", "Provide family bulk quantity discounts", "Emphasize savings in newsletters"];
    } else if (prediction === 1) {
        tagText = "👨‍👩‍👧‍👦 MODERATE SPENDER FAMILY | CORE VALUE";
        businessInsight = "These customers form the stable core of the business with reliable, steady purchases.";
        recStrategy = "Encourage upselling and cross-selling";
        recCampaign = "Targeted Product Recommendations";
        recOffer = "Loyalty Points Multiplier (3x points)";
        recPriority = "Medium";
        recActions = ["Recommend complementary items based on purchase path", "Promote loyalty club tier rewards"];
    } else { // 3
        tagText = "👤 BUDGET SINGLE | INDEPENDENT CONSCIOUS";
        businessInsight = "These customers are budget-conscious individuals who buy strictly what they need.";
        recStrategy = "Nudge towards more frequent purchases with low-barrier offers";
        recCampaign = "Flash Sales & Limited-Time In-App Notifications";
        recOffer = "10% Off Next Purchase / Free shipping on items > $15";
        recPriority = "Low";
        recActions = ["Push limited-time flash sales", "Recommend lower-tier affordable items", "Implement low-cost retargeting ads"];
    }

    tag.innerText = tagText;
    document.getElementById("res-segment-desc").innerText = businessInsight;

    // Set Churn Gauge Visuals
    const percentText = document.getElementById("res-churn-percent");
    const riskText = document.getElementById("res-churn-risk-text");
    const gaugeFill = document.getElementById("gauge-fill-circle");

    percentText.innerText = `${(churn_prob * 100).toFixed(1)}%`;
    riskText.innerText = churn_risk;

    // Calculate gauge stroke dashoffset
    // circle has radius=40, circumference = 2 * PI * r = 251.2
    const circumference = 251.2;
    const offset = circumference - (churn_prob * circumference);
    gaugeFill.style.strokeDashoffset = offset;

    // Update gauge stroke color
    let churnColor = "";
    let churnActions = [];
    let churnTriggers = "";
    
    if (churn_risk === "HIGH") {
        churnColor = "var(--danger)";
        riskText.style.color = "var(--danger)";
        gaugeFill.style.stroke = "var(--danger)";
        churnActions = ["Send immediate high-value retention offer", "Provide complimentary loyalty account points", "Trigger direct callback flag"];
        churnTriggers = "High Recency count (>60 days since purchase) or recent formal customer complain.";
    } else {
        churnColor = "var(--success)";
        riskText.style.color = "var(--success)";
        gaugeFill.style.stroke = "var(--success)";
        churnActions = ["Maintain standard engagement pipeline", "Upsell standard loyalty programs", "Periodically query for feedback"];
        churnTriggers = "Recent active purchase status within safety bounds.";
    }

    document.getElementById("res-churn-details").innerText = `${(churn_prob * 100).toFixed(0)}% Churn Probability | ${churn_risk} Risk Profile`;
    document.getElementById("res-churn-triggers").innerText = churnTriggers;

    // Render strategies
    document.getElementById("res-rec-strategy").innerText = recStrategy;
    document.getElementById("res-rec-campaign").innerText = recCampaign;
    document.getElementById("res-rec-offer").innerText = recOffer;

    const priBadge = document.getElementById("res-rec-priority");
    priBadge.innerText = recPriority;
    priBadge.className = `priority-badge ${recPriority.toLowerCase()}`;

    // Render Checklists
    const strategyList = document.getElementById("res-rec-checklist");
    strategyList.innerHTML = "";
    recActions.forEach(action => {
        const li = document.createElement("li");
        li.innerText = action;
        strategyList.appendChild(li);
    });

    const retentionList = document.getElementById("res-churn-checklist");
    retentionList.innerHTML = "";
    churnActions.forEach(action => {
        const li = document.createElement("li");
        li.innerText = action;
        retentionList.appendChild(li);
    });

    // 4. Render comparative table
    const compareBody = document.getElementById("compare-table-body");
    compareBody.innerHTML = "";

    const segmentAverages = DATASET_SUMMARY.cluster_summary[prediction];
    const compareFeatures = [
        { label: "Income", key: "Income", format: (v) => `$${v.toLocaleString()}`, unit: "$", value: income },
        { label: "Total Spending", key: "Total_Spending", format: (v) => `$${v.toLocaleString()}`, unit: "$", value: total_spending },
        { label: "Age", key: "Age", format: (v) => `${v.toFixed(0)} yrs`, unit: "yrs", value: age },
        { label: "Customer Tenure (Days)", key: "Customer_Tenure_Days", format: (v) => `${v.toFixed(0)} days`, unit: "days", value: customer_tenure },
        { label: "Total Children", key: "Total_Children", format: (v) => v.toFixed(1), unit: "", value: total_children },
        { label: "Web Purchases", key: "NumWebPurchases", format: (v) => v.toFixed(1), unit: "", value: num_web },
        { label: "Store Purchases", key: "NumStorePurchases", format: (v) => v.toFixed(1), unit: "", value: num_store },
        { label: "Web Visits/Mo", key: "NumWebVisitsMonth", format: (v) => v.toFixed(1), unit: "", value: num_web_visits },
    ];

    compareFeatures.forEach(feat => {
        const userVal = feat.value;
        const avgVal = segmentAverages[feat.key];
        const diff = userVal - avgVal;
        
        let diffClass = "";
        let diffText = "";
        
        if (diff > 0) {
            diffClass = "compare-diff-pos";
            diffText = `+${feat.format(diff)}`;
        } else if (diff < 0) {
            diffClass = "compare-diff-neg";
            diffText = `-${feat.format(Math.abs(diff))}`;
        } else {
            diffClass = "compare-diff-neutral";
            diffText = "0";
        }

        // Clean currency double format symbols
        if (feat.unit === "$") {
            if (diff > 0) diffText = `+$${Math.abs(diff).toLocaleString()}`;
            else if (diff < 0) diffText = `-$${Math.abs(diff).toLocaleString()}`;
        }

        const tr = document.createElement("tr");
        tr.innerHTML = `
            <td style="font-weight: 500;">${feat.label}</td>
            <td>${feat.format(userVal)}</td>
            <td>${feat.format(avgVal)}</td>
            <td class="${diffClass}">${diffText}</td>
        `;
        compareBody.appendChild(tr);
    });

    // 5. Reveal prediction results section
    const resultsDiv = document.getElementById("prediction-results");
    resultsDiv.style.display = "block";
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'start' });

    // 6. Trigger balloons if Premium customer
    if (prediction === 2) {
        triggerBalloons();
    }
}

// Premium customer celebration balloons animation
function triggerBalloons() {
    const canvas = document.getElementById("balloonCanvas");
    canvas.style.display = "block";
    const ctx = canvas.getContext("2d");
    
    // Set viewport dimensions
    let width = canvas.width = window.innerWidth;
    let height = canvas.height = window.innerHeight;
    
    window.addEventListener('resize', () => {
        width = canvas.width = window.innerWidth;
        height = canvas.height = window.innerHeight;
    });

    const colors = ["#f472b6", "#a78bfa", "#60a5fa", "#34d399", "#fb7185", "#fbbf24"];
    const balloons = [];

    // Create 35 balloons
    for (let i = 0; i < 35; i++) {
        balloons.push({
            x: Math.random() * width,
            y: height + Math.random() * 200 + 50,
            radius: Math.random() * 20 + 20,
            speed: Math.random() * 2.5 + 1.5,
            color: colors[Math.floor(Math.random() * colors.length)],
            sway: Math.random() * 2,
            swaySpeed: Math.random() * 0.02 + 0.01,
            phase: Math.random() * Math.PI
        });
    }

    let startTime = Date.now();

    function animate() {
        ctx.clearRect(0, 0, width, height);
        let allGone = true;

        balloons.forEach(b => {
            b.y -= b.speed;
            b.phase += b.swaySpeed;
            b.x += Math.sin(b.phase) * b.sway;

            if (b.y + b.radius > 0) {
                allGone = false;
                
                // Draw string
                ctx.beginPath();
                ctx.moveTo(b.x, b.y + b.radius);
                ctx.quadraticCurveTo(b.x + Math.sin(b.phase)*10, b.y + b.radius + 30, b.x, b.y + b.radius + 60);
                ctx.strokeStyle = "rgba(255,255,255,0.3)";
                ctx.lineWidth = 1;
                ctx.stroke();

                // Draw balloon body
                ctx.beginPath();
                ctx.ellipse(b.x, b.y, b.radius, b.radius * 1.25, 0, 0, Math.PI * 2);
                ctx.fillStyle = b.color;
                ctx.fill();

                // Highlight gloss reflection
                ctx.beginPath();
                ctx.ellipse(b.x - b.radius * 0.3, b.y - b.radius * 0.4, b.radius * 0.2, b.radius * 0.4, Math.PI / 4, 0, Math.PI * 2);
                ctx.fillStyle = "rgba(255, 255, 255, 0.4)";
                ctx.fill();

                // Draw balloon basket tie (triangle)
                ctx.beginPath();
                ctx.moveTo(b.x, b.y + b.radius * 1.2);
                ctx.lineTo(b.x - 5, b.y + b.radius * 1.25);
                ctx.lineTo(b.x + 5, b.y + b.radius * 1.25);
                ctx.closePath();
                ctx.fillStyle = b.color;
                ctx.fill();
            }
        });

        // Continue running animation for 7 seconds or until balloons go off screen
        if (!allGone && Date.now() - startTime < 7000) {
            requestAnimationFrame(animate);
        } else {
            canvas.style.display = "none";
            ctx.clearRect(0, 0, width, height);
        }
    }

    animate();
}
