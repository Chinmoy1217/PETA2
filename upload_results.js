
const snowflake = require('snowflake-sdk');
const fs = require('fs');
const csv = require('csv-parser');

const CSV_FILE = 'batch_results_processed.csv';

// Snowflake Config
const connection = snowflake.createConnection({
    account: 'COZENTUS-DATAPRACTICE',
    username: 'HACKATHON_DT',
    password: process.env.SNOWFLAKE_TOKEN,
    database: 'HACAKATHON',
    schema: 'DT_INGESTION',
    warehouse: 'cozentus_wh'
});

async function runUpload() {
    console.log("Connecting to Snowflake (via Node.js)...");

    connection.connectAsync = function () {
        return new Promise((resolve, reject) => {
            this.connect((err, conn) => {
                if (err) reject(err);
                else resolve(conn);
            });
        });
    };

    try {
        await connection.connectAsync();
        console.log("Connected to Snowflake!");

        // Create Table Name with Timestamp
        const timestamp = new Date().toISOString().replace(/[-:T.]/g, '').slice(0, 14);
        const tableName = `BATCH_RESULTS_NODE_${timestamp}`;

        const createSql = `
      CREATE TABLE IF NOT EXISTS ${tableName} (
        TRIP_ID VARCHAR, 
        POL VARCHAR, 
        POD VARCHAR, 
        MODE VARCHAR, 
        ATD TIMESTAMP_NTZ, 
        PETA_PREDICTED_DURATION FLOAT, 
        ESTIMATED_ATA TIMESTAMP_NTZ
      )
    `;

        console.log(`Creating table: ${tableName}`);
        await executeQuery(connection, createSql);

        // Read CSV and Insert
        const rows = [];
        fs.createReadStream(CSV_FILE)
            .pipe(csv())
            .on('data', (row) => rows.push(row))
            .on('end', async () => {
                console.log(`Parsed ${rows.length} rows. Uploading...`);

                try {
                    // Construct Batch Insert
                    let values = [];
                    for (const row of rows) {
                        // Ensure strings are quoted
                        values.push(`('${row.trip_id}', '${row.PolCode}', '${row.PodCode}', '${row.ModeOfTransport}', '${row.ATD}', ${row.PETA_Predicted_Duration}, '${row.Predicted_ATA}')`);
                    }

                    if (values.length > 0) {
                        const insertSql = `INSERT INTO ${tableName} VALUES ${values.join(',')}`;
                        await executeQuery(connection, insertSql);
                        console.log("✅ Insert Executed. Verifying count...");

                        // Self-Verification
                        const verifySql = `SELECT COUNT(*) as CNT FROM ${tableName}`;
                        const verifyRes = await executeQuery(connection, verifySql);
                        const count = verifyRes[0]['CNT'];

                        console.log(`🔍 VERIFICATION: Table ${tableName} has ${count} rows.`);

                        if (count == 0) {
                            console.error("❌ ERROR: Insert appeared to succeed but table is empty!");
                        } else {
                            console.log("✅ CONFIRMED: Data is in the Database.");
                        }
                    } else {
                        console.log("No data to upload.");
                    }

                } catch (err) {
                    console.error("Insert Failed:", err);
                } finally {
                    // connection.destroy(); 
                }
            });

    } catch (err) {
        console.error("Connection/Query Failed:", err.message);
    }
}

function executeQuery(conn, sql) {
    return new Promise((resolve, reject) => {
        // console.log(`[DEBUG] Executing SQL: ${sql.substring(0, 100)}...`);
        conn.execute({
            sqlText: sql,
            complete: (err, stmt, rows) => {
                if (err) reject(err);
                else resolve(rows);
            }
        });
    });
}

runUpload();
