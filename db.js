// backend/db.js
const sql = require('mssql');
require('dotenv').config();

const config = {
  user: process.env.DB_USER,           // mh_app_user
  password: process.env.DB_PASSWORD,   // Ertugrul#123
  server: process.env.DB_SERVER,       // 127.0.0.1
  database: process.env.DB_DATABASE,   // WellbeingTracker
  port: Number(process.env.DB_PORT),   // 61373
  options: {
    encrypt: false,                    // local dev only
    trustServerCertificate: true       // accept local cert
  }
};

const pool = new sql.ConnectionPool(config);
const poolConnect = pool.connect();

module.exports = {
  sql,
  pool,
  poolConnect
};




