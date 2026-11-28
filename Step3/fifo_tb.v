// Directed FIFO checks added after mutation testing: paths the full-layer runs never reach.
`timescale 1ns/1ps
module fifo_tb;
    reg clk = 0;
    always #5 clk = ~clk;
    reg reset = 1;
    integer i;

    task tick;
        begin @(posedge clk); #1; @(negedge clk); end
    endtask
    task check(input ok, input [8*48:1] what);
        begin if (ok !== 1'b1) $fatal(1, "FIFO check failed: %0s", what); end
    endtask

    reg f_rd = 0, f_wr = 0;
    reg [7:0] f_in = 0;
    wire [7:0] f_out;
    wire f_full, f_empty;
    fifo_depth64 #(.bw(8)) fifo(.rd_clk(clk), .wr_clk(clk), .reset(reset), .rd(f_rd), .wr(f_wr),
                                .in(f_in), .out(f_out), .o_full(f_full), .o_empty(f_empty));

    reg l_wr = 0;
    wire [31:0] l_out;
    wire l_full, l_ready;
    l0 #(.row(8), .bw(4)) l0_dut(.clk(clk), .reset(reset), .in(32'h12345678), .out(l_out),
                                 .rd(1'b0), .wr(l_wr), .o_full(l_full), .o_ready(l_ready));

    reg [7:0] o_wr = 0;
    reg o_rd = 0;
    wire [127:0] o_out;
    wire o_full, o_ready, o_valid;
    ofifo #(.col(8), .bw(16)) ofifo_dut(.clk(clk), .reset(reset), .in({8{16'h00ab}}), .out(o_out),
                                        .rd(o_rd), .wr(o_wr), .o_full(o_full), .o_ready(o_ready),
                                        .o_valid(o_valid));

    reg [7:0] s_wr = 0;
    reg s_rd = 0;
    reg [15:0] s_val = 0;
    wire [127:0] s_out;
    wire s_full, s_valid;
    ofifo_stream #(.COLS(8), .WIDTH(16), .DEPTH(16)) stream_dut(
        .clk(clk), .reset(reset), .wr(s_wr), .in({8{s_val}}), .rd(s_rd), .out(s_out),
        .o_full(s_full), .o_valid(s_valid));

    initial begin
        @(negedge clk);
        tick(); tick(); reset = 0; tick();

        // FIFO64: reads while empty and writes while full are ignored; order is kept.
        check(f_empty === 1'b1 && f_full === 1'b0, "fifo64 reset state");
        f_rd = 1; tick(); tick(); f_rd = 0;
        f_wr = 1;
        for (i=1; i<=64; i=i+1) begin f_in = i; tick(); end
        check(f_full, "fifo64 full after 64 writes");
        f_in = 8'hff; tick(); f_wr = 0;
        for (i=1; i<=64; i=i+1) begin
            check(f_out === i[7:0], "fifo64 order");
            f_rd = 1; tick(); f_rd = 0;
        end
        check(f_empty, "fifo64 empty after draining");

        // L0 drops ready once a row is full.
        check(l_ready, "l0 ready when empty");
        l_wr = 1; repeat (64) tick(); l_wr = 0;
        check(l_full === 1'b1 && l_ready === 1'b0, "l0 back-pressure when full");

        // Buffered OFIFO: valid needs every lane; the read request is registered.
        o_wr = 8'b0000_0001; tick(); o_wr = 0;
        check(o_valid === 1'b0, "ofifo valid needs all lanes");
        o_wr = 8'b1111_1110; tick(); o_wr = 0;
        check(o_valid, "ofifo valid with all lanes");
        o_rd = 1; tick(); o_rd = 0;
        check(o_valid, "ofifo read is registered");
        tick();
        check(o_valid === 1'b0, "ofifo popped one word");

        // Stream OFIFO: full at its depth; a write is accepted while full if a pop happens.
        s_wr = 8'hff;
        for (i=0; i<16; i=i+1) begin s_val = i; tick(); end
        s_wr = 0;
        check(s_full, "stream ofifo full at depth");
        s_val = 16; s_wr = 8'hff; s_rd = 1; tick(); s_wr = 0; s_rd = 0;
        for (i=1; i<=16; i=i+1) begin
            check(s_valid === 1'b1 && s_out[15:0] === i[15:0], "stream ofifo order after pop+push");
            s_rd = 1; tick(); s_rd = 0;
        end
        check(s_valid === 1'b0, "stream ofifo empty");
        s_rd = 1; tick(); s_rd = 0;
        check(s_valid === 1'b0, "stream ofifo ignores a read while empty");

        $display("PASS FIFO: capacity, underflow, back-pressure, lane alignment, pop+push when full");
        $finish;
    end
    initial begin #100000; $fatal(1, "FIFO timeout"); end
endmodule
