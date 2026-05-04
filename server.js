// backend/server.js
const express = require('express');
const cors = require('cors');
require('dotenv').config();
const db = require('./db');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const { requireAuth } = require('./middleware/auth');


const app = express();
const PORT = process.env.PORT || 3000;

// ---------- MIDDLEWARE ----------
app.use(cors());
app.use(express.json()); // important for JSON body parsing

const path = require('path');

// Serve frontend files
app.use(express.static(path.join(__dirname, 'public')));


// ---------- HEALTH CHECK ----------
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    message: 'Mental Health & Wellbeing API is running'
  });
});

// ---------- TEST DB CONNECTION ----------
app.get('/api/test-db', async (req, res) => {
  try {
    await db.poolConnect;
    const result = await db.pool.request().query('SELECT TOP 5 * FROM Users');
    res.json(result.recordset);
  } catch (err) {
    console.error('Database test error:', err);
    res.status(500).json({ error: 'Database test failed' });
  }
});

// ---------- AUTH: SIGNUP ----------
app.post('/api/auth/signup', async (req, res) => {
  const { fullName, email, password, consentGiven, isAnonymous } = req.body;

  // Basic validation
  if (!fullName || !email || !password) {
    return res.status(400).json({
      error: 'fullName, email and password are required.'
    });
  }

  try {
    await db.poolConnect;

    // 1) Check if email already exists
    const existing = await db.pool.request()
      .input('Email', db.sql.NVarChar(100), email)
      .query('SELECT UserID FROM Users WHERE Email = @Email');

    if (existing.recordset.length > 0) {
      return res.status(409).json({ error: 'Email already registered.' });
    }

    // 2) Hash password
    const hashed = await bcrypt.hash(password, 10);

    // 3) Insert new user
    const insertResult = await db.pool.request()
      .input('FullName', db.sql.NVarChar(100), fullName)
      .input('Email', db.sql.NVarChar(100), email)
      .input('PasswordHash', db.sql.NVarChar(255), hashed)
      .input('ConsentGiven', db.sql.Bit, consentGiven === true)
      .input('IsAnonymous', db.sql.Bit, isAnonymous === true)
      .query(`
        INSERT INTO Users (FullName, Email, PasswordHash, ConsentGiven, IsAnonymous, CreatedAt)
        OUTPUT INSERTED.UserID, INSERTED.FullName, INSERTED.Email,
               INSERTED.ConsentGiven, INSERTED.IsAnonymous, INSERTED.CreatedAt
        VALUES (@FullName, @Email, @PasswordHash, @ConsentGiven, @IsAnonymous, SYSUTCDATETIME());
      `);

    const newUser = insertResult.recordset[0];

    // 4) Create JWT token
    const token = jwt.sign(
      { userId: newUser.UserID, email: newUser.Email },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '1h' }
    );

    // 5) Send response
    res.status(201).json({
      token,
      user: {
        userId: newUser.UserID,
        fullName: newUser.FullName,
        email: newUser.Email,
        consentGiven: newUser.ConsentGiven,
        isAnonymous: newUser.IsAnonymous,
        createdAt: newUser.CreatedAt
      }
    });
  } catch (err) {
    console.error('Signup error:', err);
    res.status(500).json({ error: 'Signup failed.' });
  }
});

// ---------- START SERVER ----------
app.listen(PORT, () => {
  console.log(`Server listening on http://localhost:${PORT}`);
});

// ---------- AUTH: LOGIN ----------
app.post('/api/auth/login', async (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: "Email and password are required." });
  }

  try {
    await db.poolConnect;

    // 1) Check if email exists
    const userResult = await db.pool.request()
      .input('Email', db.sql.NVarChar(100), email)
      .query(`
        SELECT UserID, FullName, Email, PasswordHash, ConsentGiven, 
               IsAnonymous, CreatedAt
        FROM Users
        WHERE Email = @Email
      `);

    if (userResult.recordset.length === 0) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    const user = userResult.recordset[0];

    // 2) Compare password
    const valid = await bcrypt.compare(password, user.PasswordHash);
    if (!valid) {
      return res.status(401).json({ error: "Invalid email or password." });
    }

    // 3) Create token
    const token = jwt.sign(
      { userId: user.UserID, email: user.Email },
      process.env.JWT_SECRET,
      { expiresIn: process.env.JWT_EXPIRES_IN || '1h' }
    );

    // 4) Send response
    res.json({
      token,
      user: {
        userId: user.UserID,
        fullName: user.FullName,
        email: user.Email,
        consentGiven: user.ConsentGiven,
        isAnonymous: user.IsAnonymous,
        createdAt: user.CreatedAt
      }
    });

  } catch (err) {
    console.error("Login error:", err);
    res.status(500).json({ error: "Login failed." });
  }
});

// ---------- MOOD: CREATE ----------
app.post('/api/mood', requireAuth, async (req, res) => {
  const { moodScore, note, loggedAt } = req.body;

  // userId comes from the token
  const userId = req.user.userId;

  // validation
  if (moodScore === undefined || moodScore === null) {
    return res.status(400).json({ error: "moodScore is required" });
  }

  const moodNum = Number(moodScore);
  if (Number.isNaN(moodNum) || moodNum < 1 || moodNum > 10) {
    return res.status(400).json({ error: "moodScore must be a number between 1 and 10" });
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .input('MoodScore', db.sql.Int, moodNum)
      .input('Note', db.sql.NVarChar(500), note || null)
      // If loggedAt is provided use it, otherwise use current time
      .input('LoggedAt', db.sql.DateTime, loggedAt ? new Date(loggedAt) : new Date())
      .query(`
        INSERT INTO MoodLogs (UserID, MoodScore, Note, LoggedAt)
        OUTPUT INSERTED.MoodID, INSERTED.UserID, INSERTED.MoodScore, INSERTED.Note, INSERTED.LoggedAt
        VALUES (@UserID, @MoodScore, @Note, @LoggedAt);
      `);

    res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create mood error:", err);
    res.status(500).json({ error: "Failed to create mood log" });
  }
});

// ---------- MOOD: READ (all moods for logged-in user) ----------
app.get('/api/mood', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          MoodID, UserID, MoodScore, Note, LoggedAt
        FROM MoodLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get moods error:", err);
    return res.status(500).json({ error: "Failed to fetch moods" });
  }
});

// -------- SLEEP: CREATE --------
app.post('/api/sleep', requireAuth, async (req, res) => {
  const { sleepHours, sleepQuality, loggedAt } = req.body;

  // userId comes from JWT middleware
  const userId = req.user.userId;

  // Validation (light + safe)
  // sleepHours optional, but if provided must be a number (0-24 is a sensible bound)
  if (sleepHours !== undefined && sleepHours !== null) {
    const hoursNum = Number(sleepHours);
    if (Number.isNaN(hoursNum) || hoursNum < 0 || hoursNum > 24) {
      return res.status(400).json({ error: "sleepHours must be a number between 0 and 24" });
    }
  }

  // sleepQuality optional, but if provided must be a string <= 50
  if (sleepQuality !== undefined && sleepQuality !== null) {
    if (typeof sleepQuality !== "string") {
      return res.status(400).json({ error: "sleepQuality must be a string" });
    }
    if (sleepQuality.length > 50) {
      return res.status(400).json({ error: "sleepQuality must be 50 characters or less" });
    }
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .input('SleepHours', db.sql.Decimal(4, 1), sleepHours ?? null)
      .input('SleepQuality', db.sql.NVarChar(50), sleepQuality ?? null)
      .input('LoggedAt', db.sql.DateTime, loggedAt ? new Date(loggedAt) : new Date())
      .query(`
        INSERT INTO SleepLogs (UserID, SleepHours, SleepQuality, LoggedAt)
        OUTPUT
          INSERTED.SleepID,
          INSERTED.UserID,
          INSERTED.SleepHours,
          INSERTED.SleepQuality,
          INSERTED.LoggedAt
        VALUES (@UserID, @SleepHours, @SleepQuality, @LoggedAt);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create sleep error:", err);
    return res.status(500).json({ error: "Failed to create sleep log" });
  }
});


// -------- SLEEP: READ (all sleeps for logged-in user) --------
app.get('/api/sleep', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          SleepID,
          UserID,
          SleepHours,
          SleepQuality,
          LoggedAt
        FROM SleepLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get sleep error:", err);
    return res.status(500).json({ error: "Failed to fetch sleep logs" });
  }
});
// -------- JOURNAL: CREATE --------
app.post('/api/journal', requireAuth, async (req, res) => {
  const { entryText, sentimentScore, sentimentLabel, detectedCategories, loggedAt } = req.body;

  const userId = req.user.userId;

  // Basic validation (EntryText is nullable in DB, but realistically a journal entry should have text)
  if (!entryText || typeof entryText !== "string" || entryText.trim().length === 0) {
    return res.status(400).json({ error: "entryText is required" });
  }

  // Optional validations
  if (sentimentLabel !== undefined && sentimentLabel !== null) {
    if (typeof sentimentLabel !== "string") {
      return res.status(400).json({ error: "sentimentLabel must be a string" });
    }
    if (sentimentLabel.length > 20) {
      return res.status(400).json({ error: "sentimentLabel must be 20 characters or less" });
    }
  }

  if (detectedCategories !== undefined && detectedCategories !== null) {
    if (typeof detectedCategories !== "string") {
      return res.status(400).json({ error: "detectedCategories must be a string" });
    }
    if (detectedCategories.length > 255) {
      return res.status(400).json({ error: "detectedCategories must be 255 characters or less" });
    }
  }

  if (sentimentScore !== undefined && sentimentScore !== null) {
    const scoreNum = Number(sentimentScore);
    if (Number.isNaN(scoreNum)) {
      return res.status(400).json({ error: "sentimentScore must be a number" });
    }
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .input('EntryText', db.sql.NVarChar(db.sql.MAX), entryText)
      .input('SentimentScore', db.sql.Float, sentimentScore ?? null)
      .input('SentimentLabel', db.sql.NVarChar(20), sentimentLabel ?? null)
      .input('DetectedCategories', db.sql.NVarChar(255), detectedCategories ?? null)
      .input('LoggedAt', db.sql.DateTime, loggedAt ? new Date(loggedAt) : new Date())
      .query(`
        INSERT INTO JournalEntries
          (UserID, EntryText, SentimentScore, SentimentLabel, DetectedCategories, LoggedAt)
        OUTPUT
          INSERTED.JournalID,
          INSERTED.UserID,
          INSERTED.EntryText,
          INSERTED.SentimentScore,
          INSERTED.SentimentLabel,
          INSERTED.DetectedCategories,
          INSERTED.LoggedAt
        VALUES
          (@UserID, @EntryText, @SentimentScore, @SentimentLabel, @DetectedCategories, @LoggedAt);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create journal error:", err);
    return res.status(500).json({ error: "Failed to create journal entry" });
  }
});


// -------- JOURNAL: READ (all journal entries for logged-in user) --------
app.get('/api/journal', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          JournalID,
          UserID,
          EntryText,
          SentimentScore,
          SentimentLabel,
          DetectedCategories,
          LoggedAt
        FROM JournalEntries
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get journal error:", err);
    return res.status(500).json({ error: "Failed to fetch journal entries" });
  }
});
// -------- ACTIVITY: CREATE --------
app.post('/api/activity', requireAuth, async (req, res) => {
  const { activityType, durationMinutes, loggedAt } = req.body;
  const userId = req.user.userId;

  // Validation (light but helpful)
  if (activityType !== undefined && activityType !== null) {
    if (typeof activityType !== "string") {
      return res.status(400).json({ error: "activityType must be a string" });
    }
    if (activityType.length > 100) {
      return res.status(400).json({ error: "activityType must be 100 characters or less" });
    }
  }

  if (durationMinutes !== undefined && durationMinutes !== null) {
    const dNum = Number(durationMinutes);
    if (!Number.isInteger(dNum) || dNum < 0 || dNum > 1440) {
      return res.status(400).json({ error: "durationMinutes must be an integer between 0 and 1440" });
    }
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .input('ActivityType', db.sql.NVarChar(100), activityType ?? null)
      .input('DurationMinutes', db.sql.Int, durationMinutes ?? null)
      .input('LoggedAt', db.sql.DateTime, loggedAt ? new Date(loggedAt) : new Date())
      .query(`
        INSERT INTO ActivityLogs (UserID, ActivityType, DurationMinutes, LoggedAt)
        OUTPUT
          INSERTED.ActivityID,
          INSERTED.UserID,
          INSERTED.ActivityType,
          INSERTED.DurationMinutes,
          INSERTED.LoggedAt
        VALUES (@UserID, @ActivityType, @DurationMinutes, @LoggedAt);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create activity error:", err);
    return res.status(500).json({ error: "Failed to create activity log" });
  }
});


// -------- ACTIVITY: READ (all activities for logged-in user) --------
app.get('/api/activity', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input('UserID', db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          ActivityID,
          UserID,
          ActivityType,
          DurationMinutes,
          LoggedAt
        FROM ActivityLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get activity error:", err);
    return res.status(500).json({ error: "Failed to fetch activity logs" });
  }
});

// -------- RECOMMENDATIONS: CREATE (manual for now) --------
app.post('/api/recommendations', requireAuth, async (req, res) => {
  const { recommendationText, generatedBy, accepted, createdAt } = req.body;
  const userId = req.user.userId;

  if (recommendationText !== undefined && recommendationText !== null) {
    if (typeof recommendationText !== "string") {
      return res.status(400).json({ error: "recommendationText must be a string" });
    }
    if (recommendationText.length > 255) {
      return res.status(400).json({ error: "recommendationText must be 255 characters or less" });
    }
  }

  if (generatedBy !== undefined && generatedBy !== null) {
    if (typeof generatedBy !== "string") {
      return res.status(400).json({ error: "generatedBy must be a string" });
    }
    if (generatedBy.length > 50) {
      return res.status(400).json({ error: "generatedBy must be 50 characters or less" });
    }
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .input("RecommendationText", db.sql.NVarChar(255), recommendationText ?? null)
      .input("GeneratedBy", db.sql.NVarChar(50), generatedBy ?? "manual")
      .input("Accepted", db.sql.Bit, accepted ?? null)
      .input("CreatedAt", db.sql.DateTime, createdAt ? new Date(createdAt) : new Date())
      .query(`
        INSERT INTO Recommendations (UserID, RecommendationText, GeneratedBy, Accepted, CreatedAt)
        OUTPUT
          INSERTED.RecID,
          INSERTED.UserID,
          INSERTED.RecommendationText,
          INSERTED.GeneratedBy,
          INSERTED.Accepted,
          INSERTED.CreatedAt
        VALUES (@UserID, @RecommendationText, @GeneratedBy, @Accepted, @CreatedAt);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create recommendation error:", err);
    return res.status(500).json({ error: "Failed to create recommendation" });
  }
});


// -------- RECOMMENDATIONS: READ --------
app.get('/api/recommendations', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          RecID,
          UserID,
          RecommendationText,
          GeneratedBy,
          Accepted,
          CreatedAt
        FROM Recommendations
        WHERE UserID = @UserID
        ORDER BY CreatedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get recommendations error:", err);
    return res.status(500).json({ error: "Failed to fetch recommendations" });
  }
});


// -------- RECOMMENDATIONS: ACCEPT --------
app.patch('/api/recommendations/:recId/accept', requireAuth, async (req, res) => {
  const userId = req.user.userId;
  const recId = Number(req.params.recId);

  if (!Number.isInteger(recId)) {
    return res.status(400).json({ error: "recId must be an integer" });
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("RecID", db.sql.Int, recId)
      .input("UserID", db.sql.Int, userId)
      .query(`
        UPDATE Recommendations
        SET Accepted = 1
        OUTPUT
          INSERTED.RecID,
          INSERTED.UserID,
          INSERTED.RecommendationText,
          INSERTED.GeneratedBy,
          INSERTED.Accepted,
          INSERTED.CreatedAt
        WHERE RecID = @RecID AND UserID = @UserID;
      `);

    if (result.recordset.length === 0) {
      return res.status(404).json({ error: "Recommendation not found" });
    }

    return res.json(result.recordset[0]);
  } catch (err) {
    console.error("Accept recommendation error:", err);
    return res.status(500).json({ error: "Failed to accept recommendation" });
  }
});

// -------- ALERTS: CREATE (manual for now) --------
app.post('/api/alerts', requireAuth, async (req, res) => {
  const { alertType, alertMessage, acknowledged, createdAt } = req.body;
  const userId = req.user.userId;

  if (alertType !== undefined && alertType !== null) {
    if (typeof alertType !== "string") {
      return res.status(400).json({ error: "alertType must be a string" });
    }
    if (alertType.length > 100) {
      return res.status(400).json({ error: "alertType must be 100 characters or less" });
    }
  }

  if (alertMessage !== undefined && alertMessage !== null) {
    if (typeof alertMessage !== "string") {
      return res.status(400).json({ error: "alertMessage must be a string" });
    }
    if (alertMessage.length > 255) {
      return res.status(400).json({ error: "alertMessage must be 255 characters or less" });
    }
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .input("AlertType", db.sql.NVarChar(100), alertType ?? null)
      .input("AlertMessage", db.sql.NVarChar(255), alertMessage ?? null)
      .input("Acknowledged", db.sql.Bit, acknowledged ?? 0)
      .input("CreatedAt", db.sql.DateTime, createdAt ? new Date(createdAt) : new Date())
      .query(`
        INSERT INTO Alerts (UserID, AlertType, AlertMessage, Acknowledged, CreatedAt)
        OUTPUT
          INSERTED.AlertID,
          INSERTED.UserID,
          INSERTED.AlertType,
          INSERTED.AlertMessage,
          INSERTED.Acknowledged,
          INSERTED.CreatedAt
        VALUES (@UserID, @AlertType, @AlertMessage, @Acknowledged, @CreatedAt);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create alert error:", err);
    return res.status(500).json({ error: "Failed to create alert" });
  }
});


// -------- ALERTS: READ --------
app.get('/api/alerts', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (200)
          AlertID,
          UserID,
          AlertType,
          AlertMessage,
          Acknowledged,
          CreatedAt
        FROM Alerts
        WHERE UserID = @UserID
        ORDER BY CreatedAt DESC;
      `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get alerts error:", err);
    return res.status(500).json({ error: "Failed to fetch alerts" });
  }
});


// -------- ALERTS: ACKNOWLEDGE --------
app.patch('/api/alerts/:alertId/ack', requireAuth, async (req, res) => {
  const userId = req.user.userId;
  const alertId = Number(req.params.alertId);

  if (!Number.isInteger(alertId)) {
    return res.status(400).json({ error: "alertId must be an integer" });
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("AlertID", db.sql.Int, alertId)
      .input("UserID", db.sql.Int, userId)
      .query(`
        UPDATE Alerts
        SET Acknowledged = 1
        OUTPUT
          INSERTED.AlertID,
          INSERTED.UserID,
          INSERTED.AlertType,
          INSERTED.AlertMessage,
          INSERTED.Acknowledged,
          INSERTED.CreatedAt
        WHERE AlertID = @AlertID AND UserID = @UserID;
      `);

    if (result.recordset.length === 0) {
      return res.status(404).json({ error: "Alert not found" });
    }

    return res.json(result.recordset[0]);
  } catch (err) {
    console.error("Acknowledge alert error:", err);
    return res.status(500).json({ error: "Failed to acknowledge alert" });
  }
});

// -------- MODEL RUNS: CREATE --------
app.post('/api/modelruns', requireAuth, async (req, res) => {
  const { modelName, version, accuracy, f1Score, trainedOn, notes } = req.body;

  // Optional validations
  if (modelName !== undefined && modelName !== null) {
    if (typeof modelName !== "string" || modelName.length > 100) {
      return res.status(400).json({ error: "modelName must be a string up to 100 characters" });
    }
  }

  if (version !== undefined && version !== null) {
    if (typeof version !== "string" || version.length > 20) {
      return res.status(400).json({ error: "version must be a string up to 20 characters" });
    }
  }

  if (notes !== undefined && notes !== null) {
    if (typeof notes !== "string" || notes.length > 255) {
      return res.status(400).json({ error: "notes must be a string up to 255 characters" });
    }
  }

  if (accuracy !== undefined && accuracy !== null && Number.isNaN(Number(accuracy))) {
    return res.status(400).json({ error: "accuracy must be a number" });
  }

  if (f1Score !== undefined && f1Score !== null && Number.isNaN(Number(f1Score))) {
    return res.status(400).json({ error: "f1Score must be a number" });
  }

  try {
    await db.poolConnect;

    const result = await db.pool.request()
      .input("ModelName", db.sql.NVarChar(100), modelName ?? null)
      .input("Version", db.sql.NVarChar(20), version ?? null)
      .input("Accuracy", db.sql.Float, accuracy ?? null)
      .input("F1Score", db.sql.Float, f1Score ?? null)
      .input("TrainedOn", db.sql.DateTime, trainedOn ? new Date(trainedOn) : new Date())
      .input("Notes", db.sql.NVarChar(255), notes ?? null)
      .query(`
        INSERT INTO ModelRuns (ModelName, Version, Accuracy, F1Score, TrainedOn, Notes)
        OUTPUT
          INSERTED.ModelID,
          INSERTED.ModelName,
          INSERTED.Version,
          INSERTED.Accuracy,
          INSERTED.F1Score,
          INSERTED.TrainedOn,
          INSERTED.Notes
        VALUES (@ModelName, @Version, @Accuracy, @F1Score, @TrainedOn, @Notes);
      `);

    return res.status(201).json(result.recordset[0]);
  } catch (err) {
    console.error("Create model run error:", err);
    return res.status(500).json({ error: "Failed to create model run" });
  }
});


// -------- MODEL RUNS: READ --------
app.get('/api/modelruns', requireAuth, async (req, res) => {
  try {
    await db.poolConnect;

    const result = await db.pool.request().query(`
      SELECT TOP (200)
        ModelID,
        ModelName,
        Version,
        Accuracy,
        F1Score,
        TrainedOn,
        Notes
      FROM ModelRuns
      ORDER BY TrainedOn DESC;
    `);

    return res.json(result.recordset);
  } catch (err) {
    console.error("Get model runs error:", err);
    return res.status(500).json({ error: "Failed to fetch model runs" });
  }
});
// -------- STEP 5: RULE-BASED "AI" RUN --------
app.post('/api/ml/run', requireAuth, async (req, res) => {
  const userId = req.user.userId;

  try {
    await db.poolConnect;

    // 1) Pull recent data (last 14 days)
    const moodsRes = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (14) MoodScore, LoggedAt
        FROM MoodLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    const sleepRes = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (14) SleepHours, SleepQuality, LoggedAt
        FROM SleepLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    const activityRes = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (14) ActivityType, DurationMinutes, LoggedAt
        FROM ActivityLogs
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    const journalRes = await db.pool.request()
      .input("UserID", db.sql.Int, userId)
      .query(`
        SELECT TOP (10) JournalID, SentimentScore, SentimentLabel, LoggedAt
        FROM JournalEntries
        WHERE UserID = @UserID
        ORDER BY LoggedAt DESC;
      `);

    const moods = moodsRes.recordset;
    const sleeps = sleepRes.recordset;
    const activities = activityRes.recordset;
    const journals = journalRes.recordset;

    // Helper calculations
    const avg = (arr) => arr.length ? (arr.reduce((a,b)=>a+b,0) / arr.length) : null;

    const avgMood = avg(moods.map(m => Number(m.MoodScore)).filter(n => !Number.isNaN(n)));
    const avgSleep = avg(sleeps.map(s => Number(s.SleepHours)).filter(n => !Number.isNaN(n)));

    const totalActivityMinutes7Days = activities
      .filter(a => {
        const d = new Date(a.LoggedAt);
        const now = new Date();
        return (now - d) <= (7 * 24 * 60 * 60 * 1000);
      })
      .map(a => Number(a.DurationMinutes))
      .filter(n => !Number.isNaN(n))
      .reduce((sum, n) => sum + n, 0);

    const negativeJournalCount = journals.filter(j => (j.SentimentLabel || "").toLowerCase() === "negative").length;

    // 2) Rule-based decisions
    const recommendations = [];
    const alerts = [];

    // Rule 1: Low mood trend
    if (avgMood !== null && avgMood < 4) {
      recommendations.push({
        text: "Your average mood has been low recently. Try a short walk, journaling, or a relaxing routine today.",
        by: "rule-based-v1"
      });
      alerts.push({
        type: "LowMoodTrend",
        message: "Mood has been low on average. Consider self-care actions or reaching out to someone you trust."
      });
    }

    // Rule 2: Low sleep
    if (avgSleep !== null && avgSleep < 6) {
      recommendations.push({
        text: "Your sleep average is below 6 hours. Consider a consistent bedtime and reducing screen time before sleep.",
        by: "rule-based-v1"
      });
      alerts.push({
        type: "LowSleep",
        message: "Sleep is consistently low. Improving sleep can strongly support mood and wellbeing."
      });
    }

    // Rule 3: Low activity
    if (totalActivityMinutes7Days < 60) {
      recommendations.push({
        text: "Activity levels look low this week. Even a 10–20 minute walk can support wellbeing.",
        by: "rule-based-v1"
      });
    }

    // Rule 4: Negative journal pattern (if using sentiment labels)
    if (negativeJournalCount >= 3) {
      alerts.push({
        type: "NegativeJournalPattern",
        message: "Several recent journal entries look negative. Consider grounding techniques or support resources."
      });
    }

    // If no rules triggered, still provide a positive recommendation
    if (recommendations.length === 0) {
      recommendations.push({
        text: "You’re maintaining a steady pattern. Keep logging your mood, sleep, and activity for ongoing insights.",
        by: "rule-based-v1"
      });
    }

    // 3) Insert outputs into DB
    const createdRecs = [];
    for (const r of recommendations) {
      const recInsert = await db.pool.request()
        .input("UserID", db.sql.Int, userId)
        .input("RecommendationText", db.sql.NVarChar(255), r.text)
        .input("GeneratedBy", db.sql.NVarChar(50), r.by)
        .input("Accepted", db.sql.Bit, 0)
        .input("CreatedAt", db.sql.DateTime, new Date())
        .query(`
          INSERT INTO Recommendations (UserID, RecommendationText, GeneratedBy, Accepted, CreatedAt)
          OUTPUT INSERTED.*
          VALUES (@UserID, @RecommendationText, @GeneratedBy, @Accepted, @CreatedAt);
        `);
      createdRecs.push(recInsert.recordset[0]);
    }

    const createdAlerts = [];
    for (const a of alerts) {
      const alertInsert = await db.pool.request()
        .input("UserID", db.sql.Int, userId)
        .input("AlertType", db.sql.NVarChar(100), a.type)
        .input("AlertMessage", db.sql.NVarChar(255), a.message)
        .input("Acknowledged", db.sql.Bit, 0)
        .input("CreatedAt", db.sql.DateTime, new Date())
        .query(`
          INSERT INTO Alerts (UserID, AlertType, AlertMessage, Acknowledged, CreatedAt)
          OUTPUT INSERTED.*
          VALUES (@UserID, @AlertType, @AlertMessage, @Acknowledged, @CreatedAt);
        `);
      createdAlerts.push(alertInsert.recordset[0]);
    }

    // 4) Log into ModelRuns
    const modelRun = await db.pool.request()
      .input("ModelName", db.sql.NVarChar(100), "RuleBasedInsights")
      .input("Version", db.sql.NVarChar(20), "v1")
      .input("Accuracy", db.sql.Float, null)
      .input("F1Score", db.sql.Float, null)
      .input("TrainedOn", db.sql.DateTime, new Date())
      .input("Notes", db.sql.NVarChar(255), `Rules executed for userId=${userId}. avgMood=${avgMood ?? "n/a"}, avgSleep=${avgSleep ?? "n/a"}`)
      .query(`
        INSERT INTO ModelRuns (ModelName, Version, Accuracy, F1Score, TrainedOn, Notes)
        OUTPUT INSERTED.*
        VALUES (@ModelName, @Version, @Accuracy, @F1Score, @TrainedOn, @Notes);
      `);

    return res.json({
      message: "AI run complete",
      avgMood,
      avgSleep,
      totalActivityMinutes7Days,
      negativeJournalCount,
      recommendationsCreated: createdRecs.length,
      alertsCreated: createdAlerts.length,
      modelRun: modelRun.recordset[0],
      createdRecs,
      createdAlerts
    });

  } catch (err) {
    console.error("ML run error:", err);
    return res.status(500).json({ error: "Failed to run AI logic" });
  }
});
